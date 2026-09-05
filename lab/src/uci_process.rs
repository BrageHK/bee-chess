//! A typed UCI client speaking directly to a spawned engine
//! subprocess's stdin/stdout -- the server-side counterpart to
//! `uci_relay` (which only relays raw bytes to a browser). This is
//! what lets `Game`'s automatic play loop (#69/67b, slice 69b) ask an
//! engine for a move itself, instead of requiring a human to relay one
//! in via `POST /api/games/:id/moves`.
//!
//! Deliberately minimal: just enough of the UCI protocol to run one
//! engine through one game (`uci`/`isready` handshake, `position`,
//! `go movetime`, read `bestmove`). No `setoption`, no `ponder`, no
//! concurrent search cancellation -- those are real gaps (Stockfish's
//! strength/Elo limiting in particular isn't wired up here the way the
//! frontend's direct WebSocket connection already does it), noted as
//! follow-up work rather than solved now, since proving the automatic
//! play loop itself is this slice's actual goal.

use std::path::Path;
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

/// Which direction a line of raw UCI traffic went -- passed to
/// `UciProcess`'s `on_line` callback so a caller building an event
/// stream (see `game::run_engine_loop`, #69's 69c-1a) can tag it
/// correctly without needing to track direction itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UciDirection {
    Sent,
    Received,
}

/// A running engine process, mid-UCI-conversation.
///
/// `on_line`, if set, is called with every line sent to or received
/// from the process -- `info` telemetry included, not just the lines
/// this type's own methods act on (`uciok`/`readyok`/`bestmove`). This
/// is what lets a caller mirror the exact same raw UCI traffic a
/// direct browser connection to the engine would see (`uci_relay`'s
/// job for the old bridge/lab relay), now that the server itself is
/// the one actually talking to the process.
pub struct UciProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    on_line: Option<OnLine>,
}

/// A callback invoked with every line `UciProcess` sends or receives --
/// see its struct docs. Factored into its own alias since the bare
/// trait-object type reads as noise wherever it appears (clippy's
/// `type_complexity` lint agrees).
pub type OnLine = Box<dyn Fn(UciDirection, &str) + Send + Sync>;

#[derive(Debug)]
pub enum UciProcessError {
    Spawn(std::io::Error),
    Io(std::io::Error),
    /// The process exited (or its stdout closed) before producing the
    /// reply we were waiting for.
    ProcessExited,
}

impl std::fmt::Display for UciProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UciProcessError::Spawn(err) => write!(f, "failed to spawn engine process: {err}"),
            UciProcessError::Io(err) => write!(f, "I/O error talking to engine process: {err}"),
            UciProcessError::ProcessExited => {
                write!(f, "engine process exited before replying")
            }
        }
    }
}

impl std::error::Error for UciProcessError {}

impl UciProcess {
    /// Spawns `argv[0]` (with `argv[1..]` as arguments) in `cwd` and
    /// completes the `uci`/`isready` handshake, discarding every line
    /// in between (`id name`, `option ...`) for this type's own
    /// purposes -- this slice doesn't need any of it beyond
    /// confirmation the engine is alive and ready. `on_line`, if given,
    /// still sees every line regardless (see the struct docs).
    pub async fn spawn(
        argv: &[String],
        cwd: &Path,
        on_line: Option<OnLine>,
    ) -> Result<Self, UciProcessError> {
        let (program, args) = argv.split_first().expect("argv must be non-empty");

        let mut child = Command::new(program)
            .args(args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(UciProcessError::Spawn)?;

        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout")).lines();

        let mut process = UciProcess {
            child,
            stdin,
            stdout,
            on_line,
        };

        process.send("uci").await?;
        process.wait_for("uciok").await?;
        process.send("isready").await?;
        process.wait_for("readyok").await?;

        Ok(process)
    }

    /// Sets the position to `startpos` plus `moves` (UCI long
    /// algebraic notation, e.g. `["e2e4", "e7e5"]`), then asks for a
    /// move with a `budget_ms` time limit, returning the move it
    /// picks (or `None` for `bestmove 0000`, meaning no legal move --
    /// checkmate/stalemate, which the engine itself detected).
    pub async fn best_move(
        &mut self,
        moves: &[String],
        budget_ms: u64,
    ) -> Result<Option<String>, UciProcessError> {
        let position_cmd = if moves.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };
        self.send(&position_cmd).await?;
        self.send(&format!("go movetime {budget_ms}")).await?;

        loop {
            let line = self.read_line().await?;
            let Some(rest) = line.strip_prefix("bestmove") else {
                continue; // ignore info lines etc. -- see module docs
            };
            let mv = rest.split_whitespace().next().unwrap_or("0000");
            return Ok(if mv == "0000" {
                None
            } else {
                Some(mv.to_string())
            });
        }
    }

    async fn send(&mut self, line: &str) -> Result<(), UciProcessError> {
        if let Some(on_line) = &self.on_line {
            on_line(UciDirection::Sent, line);
        }
        self.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(UciProcessError::Io)?;
        self.stdin
            .write_all(b"\n")
            .await
            .map_err(UciProcessError::Io)?;
        self.stdin.flush().await.map_err(UciProcessError::Io)
    }

    async fn read_line(&mut self) -> Result<String, UciProcessError> {
        match self.stdout.next_line().await {
            Ok(Some(line)) => {
                if let Some(on_line) = &self.on_line {
                    on_line(UciDirection::Received, &line);
                }
                Ok(line)
            }
            Ok(None) => Err(UciProcessError::ProcessExited),
            Err(err) => Err(UciProcessError::Io(err)),
        }
    }

    async fn wait_for(&mut self, expected: &str) -> Result<(), UciProcessError> {
        loop {
            let line = self.read_line().await?;
            if line.trim() == expected {
                return Ok(());
            }
        }
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        // `kill_on_drop(true)` (set at spawn) handles actually killing
        // the process; nothing else to clean up here. This impl exists
        // only so future fields don't have to remember to add one.
        let _ = &self.child;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `UciProcess` against a minimal hand-rolled "engine" script
    /// (a `sh` one-liner) rather than a real chess engine binary, so
    /// these tests don't depend on Stockfish/Bee being built in this
    /// environment -- they exercise the protocol-driving logic itself,
    /// not any particular engine's actual chess strength.
    fn fake_engine_argv(script: &str) -> Vec<String> {
        vec!["sh".to_string(), "-c".to_string(), script.to_string()]
    }

    #[tokio::test]
    async fn spawn_completes_the_handshake_against_a_well_behaved_fake_engine() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "id name fake"; echo "uciok"
            read _; echo "readyok"
            "#,
        );
        let result = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None).await;
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[tokio::test]
    async fn best_move_returns_the_move_after_a_go() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            read _; read _; echo "bestmove e2e4"
            "#,
        );
        let mut process = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None)
            .await
            .expect("handshake should succeed");

        let mv = process
            .best_move(&[], 100)
            .await
            .expect("should get a bestmove");
        assert_eq!(mv, Some("e2e4".to_string()));
    }

    #[tokio::test]
    async fn best_move_skips_info_lines_before_bestmove() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            read _; read _
            echo "info depth 1 score cp 10 pv e2e4"
            echo "bestmove e2e4"
            "#,
        );
        let mut process = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None)
            .await
            .expect("handshake should succeed");

        let mv = process
            .best_move(&[], 100)
            .await
            .expect("should get a bestmove");
        assert_eq!(mv, Some("e2e4".to_string()));
    }

    #[tokio::test]
    async fn best_move_none_for_bestmove_0000() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            read _; read _; echo "bestmove 0000"
            "#,
        );
        let mut process = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None)
            .await
            .expect("handshake should succeed");

        let mv = process
            .best_move(&[], 100)
            .await
            .expect("should get a bestmove");
        assert_eq!(mv, None);
    }

    #[tokio::test]
    async fn spawn_of_a_nonexistent_binary_is_a_spawn_error() {
        let argv = vec!["/no/such/binary/exists".to_string()];
        let result = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None).await;
        assert!(matches!(result, Err(UciProcessError::Spawn(_))));
    }

    #[tokio::test]
    async fn best_move_errors_if_the_process_exits_before_replying() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            exit 0
            "#,
        );
        let mut process = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None)
            .await
            .expect("handshake should succeed");

        let result = process.best_move(&[], 100).await;
        assert!(matches!(result, Err(UciProcessError::ProcessExited)));
    }

    #[tokio::test]
    async fn on_line_sees_every_sent_and_received_line_including_info() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            read _; read _
            echo "info depth 1 score cp 10 pv e2e4"
            echo "bestmove e2e4"
            "#,
        );
        let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let lines_for_callback = lines.clone();
        let mut process = UciProcess::spawn(
            &argv,
            std::env::temp_dir().as_path(),
            Some(Box::new(move |direction, line: &str| {
                lines_for_callback
                    .lock()
                    .unwrap()
                    .push((direction, line.to_string()));
            })),
        )
        .await
        .expect("handshake should succeed");

        process
            .best_move(&[], 100)
            .await
            .expect("should get a bestmove");

        let captured = lines.lock().unwrap();
        assert!(
            captured
                .iter()
                .any(|(dir, line)| *dir == UciDirection::Sent && line == "uci"),
            "should have seen the sent 'uci' line: {captured:?}"
        );
        assert!(
            captured
                .iter()
                .any(|(dir, line)| *dir == UciDirection::Received
                    && line.starts_with("info depth 1")),
            "should have seen the received info line, not just bestmove: {captured:?}"
        );
        assert!(
            captured
                .iter()
                .any(|(dir, line)| *dir == UciDirection::Received && line == "bestmove e2e4"),
            "should have seen the received bestmove line: {captured:?}"
        );
    }
}
