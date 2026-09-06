//! A typed UCI client speaking directly to a spawned engine
//! subprocess's stdin/stdout. This is what lets `Game`'s automatic
//! play loop (#69/67b, slice 69b) ask an engine for a move itself,
//! instead of requiring a human to relay one in via
//! `POST /api/games/:id/moves`.
//!
//! Deliberately minimal: just enough of the UCI protocol to run one
//! engine through one game (`uci`/`isready` handshake, `setoption`,
//! `debug`, `position`, `go movetime`, read `bestmove`). Still no
//! `ponder` or concurrent search cancellation -- real gaps, noted as
//! follow-up work rather than solved now, since proving the automatic
//! play loop itself was this module's original goal (`setoption`/
//! `debug` were added afterward, for #69's 69c-1b prerequisite, once a
//! lab-driven game needed to configure Stockfish's Elo or Bee's debug
//! output the way a direct browser connection already could).

use std::path::Path;
use std::process::Stdio;

use serde::{Deserialize, Serialize};
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

/// One `option` line an engine advertised during the `uci` handshake,
/// parsed into UCI's own generic type vocabulary rather than anything
/// engine-specific -- this is the whole point (see the module docs on
/// `options()`): Bee Lab, and everything downstream of it, discovers
/// what an engine supports instead of hardcoding option names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UciOption {
    Check {
        name: String,
        default: bool,
    },
    Spin {
        name: String,
        default: i64,
        min: i64,
        max: i64,
    },
    Combo {
        name: String,
        default: String,
        values: Vec<String>,
    },
    /// `button` (a fire-and-forget action, no value) intentionally has
    /// no case here: `set_option` always sends a value, and a button
    /// option isn't something an A/B experiment configures -- see
    /// `parse_option_line`'s docs on why it's the one UCI option type
    /// this type doesn't represent.
    String {
        name: String,
        default: String,
    },
}

/// Parses one `option name <name> type <type> ...` line (the exact
/// text an engine writes to stdout during the `uci` handshake) into a
/// typed `UciOption`. Returns `None` for a `button`-type option (see
/// `UciOption`'s docs) or anything malformed -- an engine advertising
/// something this parser doesn't understand shouldn't take down
/// discovery for every option it *does* understand, so `spawn` (which
/// calls this once per `option` line) simply skips a line this returns
/// `None` for rather than failing the whole handshake.
///
/// UCI's own grammar for this line is `option name <name> type <type>
/// [default <value>] [min <value>] [max <value>] [var <value>]...` --
/// `<name>` and `<default>`'s value may themselves contain spaces, so
/// this walks token-by-token rather than splitting naively, using each
/// keyword (`type`/`default`/`min`/`max`/`var`) as the next field's
/// delimiter, exactly as engines rely on GUIs already doing.
fn parse_option_line(line: &str) -> Option<UciOption> {
    let rest = line.strip_prefix("option name ")?;
    let tokens: Vec<&str> = rest.split_whitespace().collect();

    let type_index = tokens.iter().position(|&t| t == "type")?;
    let name = tokens[..type_index].join(" ");
    if name.is_empty() {
        return None;
    }
    let kind = *tokens.get(type_index + 1)?;

    let field = |keyword: &str| -> Option<String> {
        let start = tokens.iter().position(|&t| t == keyword)? + 1;
        let stop_words = ["default", "min", "max", "var"];
        let end = tokens[start..]
            .iter()
            .position(|t| stop_words.contains(t))
            .map_or(tokens.len(), |offset| start + offset);
        if start >= end {
            return None;
        }
        Some(tokens[start..end].join(" "))
    };

    match kind {
        "check" => Some(UciOption::Check {
            name,
            default: field("default")?.eq_ignore_ascii_case("true"),
        }),
        "spin" => Some(UciOption::Spin {
            name,
            default: field("default")?.parse().ok()?,
            min: field("min")?.parse().ok()?,
            max: field("max")?.parse().ok()?,
        }),
        "combo" => {
            let default = field("default")?;
            let mut values = Vec::new();
            let mut search_from = type_index;
            while let Some(offset) = tokens[search_from..].iter().position(|&t| t == "var") {
                let start = search_from + offset + 1;
                let end = tokens[start..]
                    .iter()
                    .position(|&t| t == "var")
                    .map_or(tokens.len(), |o| start + o);
                if start < end {
                    values.push(tokens[start..end].join(" "));
                }
                search_from = end;
            }
            Some(UciOption::Combo {
                name,
                default,
                values,
            })
        }
        "string" => Some(UciOption::String {
            name,
            // `field` stops at the next keyword, but a string value is
            // free text and could legitimately contain a word like
            // "min" -- there is no further field after `default` for
            // a string option, so it's safe to take everything after
            // it verbatim instead.
            default: {
                let start = tokens.iter().position(|&t| t == "default")? + 1;
                tokens[start..].join(" ")
            },
        }),
        // "button" (no value) and anything unrecognized -- see this
        // function's docs.
        _ => None,
    }
}

/// A running engine process, mid-UCI-conversation.
///
/// `on_line`, if set, is called with every line sent to or received
/// from the process -- `info` telemetry included, not just the lines
/// this type's own methods act on (`uciok`/`readyok`/`bestmove`). This
/// is what lets a caller (see `game::run_engine_loop`) mirror that raw
/// UCI traffic out over `GET /ws/games/:id` (`GameEvent::Uci`),
/// exactly what a direct browser connection to the engine used to see
/// before the frontend stopped connecting to engines directly (#89).
pub struct UciProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    on_line: Option<OnLine>,
    /// Every `option` line advertised during the handshake, parsed via
    /// `parse_option_line` -- see `options()`.
    options: Vec<UciOption>,
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
            options: Vec::new(),
        };

        process.send("uci").await?;
        process.wait_for_uciok().await?;
        process.send("isready").await?;
        process.wait_for("readyok").await?;

        Ok(process)
    }

    /// Every option this engine advertised during the `uci` handshake,
    /// parsed into UCI's own generic type vocabulary (see `UciOption`)
    /// rather than anything engine-specific -- e.g. the `GET
    /// /api/engines/:name/options` endpoint returns exactly this,
    /// letting Bee Lab (and the frontend beyond it) render/offer
    /// whatever options an engine happens to support without either
    /// one knowing the option's name ahead of time.
    pub fn options(&self) -> &[UciOption] {
        &self.options
    }

    /// Sends `setoption name <name> value <value>` and waits for the
    /// engine to confirm it's still ready, so a caller can rely on the
    /// option being applied before the next `go` -- same shape as the
    /// frontend's own `UciClient.setOption` (`engine.ts`), now needed
    /// server-side too so a lab-driven game can configure Stockfish's
    /// Elo or Bee's debug output the way a direct browser connection
    /// already could (see #69's 69c-1b prerequisite).
    pub async fn set_option(&mut self, name: &str, value: &str) -> Result<(), UciProcessError> {
        self.send(&format!("setoption name {name} value {value}"))
            .await?;
        self.send("isready").await?;
        self.wait_for("readyok").await
    }

    /// Sends `debug on`/`debug off` and waits for the engine to
    /// confirm it's still ready. Same reasoning as `set_option`: this
    /// mirrors `UciClient.setDebug` on the frontend.
    pub async fn set_debug(&mut self, on: bool) -> Result<(), UciProcessError> {
        self.send(if on { "debug on" } else { "debug off" }).await?;
        self.send("isready").await?;
        self.wait_for("readyok").await
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

    /// Same as `wait_for("uciok")`, but also parses every `option`
    /// line seen along the way into `self.options` -- per UCI, an
    /// engine advertises `id`/`option` lines after `uci` and before
    /// `uciok`, never after, so this is the one place discovery needs
    /// to happen.
    async fn wait_for_uciok(&mut self) -> Result<(), UciProcessError> {
        loop {
            let line = self.read_line().await?;
            if line.trim() == "uciok" {
                return Ok(());
            }
            if let Some(option) = parse_option_line(line.trim()) {
                self.options.push(option);
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

    #[test]
    fn parses_a_check_option() {
        assert_eq!(
            parse_option_line("option name UseTT type check default true"),
            Some(UciOption::Check {
                name: "UseTT".to_string(),
                default: true,
            })
        );
    }

    #[test]
    fn parses_a_check_option_defaulting_to_false() {
        assert_eq!(
            parse_option_line("option name Ponder type check default false"),
            Some(UciOption::Check {
                name: "Ponder".to_string(),
                default: false,
            })
        );
    }

    #[test]
    fn parses_a_spin_option() {
        assert_eq!(
            parse_option_line("option name UCI_Elo type spin default 1600 min 1320 max 3190"),
            Some(UciOption::Spin {
                name: "UCI_Elo".to_string(),
                default: 1600,
                min: 1320,
                max: 3190,
            })
        );
    }

    #[test]
    fn parses_a_combo_option_with_multiple_values() {
        assert_eq!(
            parse_option_line(
                "option name Evaluator type combo default Positional var Positional var Material"
            ),
            Some(UciOption::Combo {
                name: "Evaluator".to_string(),
                default: "Positional".to_string(),
                values: vec!["Positional".to_string(), "Material".to_string()],
            })
        );
    }

    #[test]
    fn parses_a_string_option() {
        assert_eq!(
            parse_option_line("option name ModelFile type string default "),
            Some(UciOption::String {
                name: "ModelFile".to_string(),
                default: String::new(),
            })
        );
    }

    #[test]
    fn parses_a_multi_word_option_name() {
        // UCI names can contain spaces (e.g. Stockfish's real "Move
        // Overhead") -- everything before the `type` keyword is the
        // name, not just the first token.
        assert_eq!(
            parse_option_line("option name Move Overhead type spin default 10 min 0 max 5000"),
            Some(UciOption::Spin {
                name: "Move Overhead".to_string(),
                default: 10,
                min: 0,
                max: 5000,
            })
        );
    }

    #[test]
    fn button_options_are_not_represented() {
        assert_eq!(
            parse_option_line("option name Clear Hash type button"),
            None
        );
    }

    #[test]
    fn malformed_option_line_is_none() {
        assert_eq!(parse_option_line("option name Broken type"), None);
        assert_eq!(parse_option_line("not an option line at all"), None);
    }

    #[tokio::test]
    async fn spawn_captures_options_advertised_before_uciok() {
        let argv = fake_engine_argv(
            r#"
            read _
            echo "id name fake"
            echo "option name UseTT type check default true"
            echo "option name Evaluator type combo default Positional var Positional var Material"
            echo "uciok"
            read _; echo "readyok"
            "#,
        );
        let process = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None)
            .await
            .expect("handshake should succeed");

        assert_eq!(
            process.options(),
            &[
                UciOption::Check {
                    name: "UseTT".to_string(),
                    default: true,
                },
                UciOption::Combo {
                    name: "Evaluator".to_string(),
                    default: "Positional".to_string(),
                    values: vec!["Positional".to_string(), "Material".to_string()],
                },
            ]
        );
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
        // Two distinct real races can both legitimately fire here,
        // depending on exactly when the already-`exit 0`'d process's
        // pipes actually close relative to our writes: `read_line`
        // seeing stdout closed (`ProcessExited`), or `send`'s write to
        // an already-closed stdin failing first (`Io`, e.g. a broken
        // pipe). Both correctly mean "the process is gone" -- and
        // `run_engine_loop`'s caller already treats every
        // `UciProcessError` variant identically (see its `Err(err) =>
        // store.abort(...)` catch-all) -- so asserting either is the
        // right test, not just the one that happens to win the race
        // under a given system load.
        assert!(
            matches!(
                result,
                Err(UciProcessError::ProcessExited) | Err(UciProcessError::Io(_))
            ),
            "{result:?}"
        );
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

    #[tokio::test]
    async fn set_option_sends_setoption_and_waits_for_readyok() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            read line; echo "got: $line" >> /dev/null
            read _; echo "readyok"
            "#,
        );
        let mut process = UciProcess::spawn(&argv, std::env::temp_dir().as_path(), None)
            .await
            .expect("handshake should succeed");

        let result = process.set_option("UCI_Elo", "1600").await;
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[tokio::test]
    async fn set_option_captures_the_exact_setoption_line_via_on_line() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            read _
            read _; echo "readyok"
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
            .set_option("UCI_LimitStrength", "true")
            .await
            .expect("set_option should succeed");

        let captured = lines.lock().unwrap();
        assert!(
            captured.iter().any(|(dir, line)| *dir == UciDirection::Sent
                && line == "setoption name UCI_LimitStrength value true"),
            "should have sent the exact setoption line: {captured:?}"
        );
    }

    #[tokio::test]
    async fn set_debug_sends_debug_on_and_waits_for_readyok() {
        let argv = fake_engine_argv(
            r#"
            read _; echo "uciok"
            read _; echo "readyok"
            read _
            read _; echo "readyok"
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

        let result = process.set_debug(true).await;
        assert!(result.is_ok(), "{:?}", result.err());

        let captured = lines.lock().unwrap();
        assert!(
            captured
                .iter()
                .any(|(dir, line)| *dir == UciDirection::Sent && line == "debug on"),
            "should have sent 'debug on': {captured:?}"
        );
    }
}
