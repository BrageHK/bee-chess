//! Spawns a UCI engine subprocess per WebSocket connection and relays
//! raw UCI lines to/from its stdin/stdout, unmodified.
//!
//! This is a straight Rust port of `bridge/server.py`'s `make_handler` --
//! see #68 (67a). Same behavior, same reasoning throughout, including
//! `watch_for_exit`'s crash-visibility fix: if the engine process dies
//! (missing binary, bad checkpoint path, whatever) before ever printing
//! a UCI reply, neither the stdout-pump task (its read just returns EOF)
//! nor the "relay whatever the browser sends" loop notices on their own
//! -- the browser would be left waiting forever for a reply that will
//! never come, with no signal that the socket is dead. `watch_for_exit`
//! actively closes the socket once the process exits, after sending
//! whatever it printed to stderr as one diagnostic line, so the reason
//! is visible instead of a silent hang.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

/// One engine this server knows how to spawn a process for: `argv[0]`
/// plus any fixed arguments (e.g. `["/path/to/stockfish"]`), and the
/// working directory to spawn it in.
#[derive(Debug, Clone)]
pub struct EngineSpec {
    pub argv: Vec<String>,
    pub cwd: PathBuf,
}

/// Builds a one-route `Router` that upgrades any request at `path` to
/// a WebSocket and relays it to a fresh process spawned from `spec`,
/// per `relay`'s docs. Exists mainly so tests can exercise the real
/// WebSocket-upgrade path (via a real client connecting to a real
/// bound server) instead of only unit-testing `relay` in isolation.
pub fn route(path: &str, spec: EngineSpec) -> Router {
    Router::new().route(path, get(handler)).with_state(spec)
}

async fn handler(ws: WebSocketUpgrade, State(spec): State<EngineSpec>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        relay(socket, &spec.argv, &spec.cwd).await;
    })
}

/// Spawns `argv[0]` (with `argv[1..]` as arguments) in `cwd` and relays
/// `socket` <-> the process's stdin/stdout line by line until either
/// side closes, exactly like `bridge/server.py`'s per-connection
/// handler.
pub async fn relay(socket: WebSocket, argv: &[String], cwd: &Path) {
    let Some((program, args)) = argv.split_first() else {
        tracing::error!("relay called with an empty argv");
        return;
    };

    let mut child = match Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            tracing::error!("failed to spawn {program:?}: {err}");
            return;
        }
    };

    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    let (mut ws_sink, mut ws_stream) = socket.split();

    // Pumps the child's stdout to the browser, one line at a time, same
    // shape as the Python bridge's `pump()`. Ends (returns) once the
    // child closes stdout, which happens on exit -- that alone doesn't
    // close the WebSocket; `watch_for_exit` below is what actually does
    // that, since a clean pump-loop exit here is also what happens on
    // an entirely healthy `quit`.
    let pump = async move {
        let mut lines = BufReader::new(stdout).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    if ws_sink.send(Message::Text(line.into())).await.is_err() {
                        break; // browser side gone
                    }
                }
                Ok(None) => break, // stdout closed (process exited)
                Err(err) => {
                    tracing::warn!("error reading engine stdout: {err}");
                    break;
                }
            }
        }
        ws_sink
    };

    // Relays whatever the browser sends straight to the child's stdin,
    // same shape as the Python bridge's `async for msg in ws` loop.
    let forward = async move {
        while let Some(Ok(msg)) = ws_stream.next().await {
            if let Message::Text(text) = msg {
                if stdin.write_all(text.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.write_all(b"\n").await.is_err() {
                    break;
                }
                if stdin.flush().await.is_err() {
                    break;
                }
            }
        }
    };

    tokio::pin!(pump);
    tokio::pin!(forward);

    tokio::select! {
        mut ws_sink = &mut pump => {
            // Process exited (or the browser vanished) -- report why,
            // then make sure the socket actually closes so the browser
            // knows, instead of hanging on a reply that will never come.
            let reason = exit_reason(&mut child, stderr).await;
            let _ = ws_sink.send(Message::Text(format!("info string engine process exited: {reason}").into())).await;
            let _ = ws_sink.close().await;
        }
        () = &mut forward => {
            // Browser closed its side (navigated away, refreshed, etc.)
            // -- nothing more to relay; the child is killed via
            // `kill_on_drop` when `child` drops at the end of this
            // function.
        }
    }
}

/// Waits for `child` to actually exit (it may already have, if we're
/// here because stdout closed) and returns a one-line reason: the last
/// line of stderr if it printed anything, otherwise just the exit
/// code. Mirrors the Python bridge's `watch_for_exit` reasoning
/// exactly -- the last stderr line is usually the actual Python/Rust
/// panic/error message, which is far more useful than a bare exit code.
async fn exit_reason(child: &mut Child, mut stderr: tokio::process::ChildStderr) -> String {
    let mut stderr_bytes = Vec::new();
    let _ = stderr.read_to_end(&mut stderr_bytes).await;
    let stderr_text = String::from_utf8_lossy(&stderr_bytes);
    let last_line = stderr_text
        .lines()
        .next_back()
        .map(str::trim)
        .filter(|l| !l.is_empty());

    match last_line {
        Some(line) => line.to_string(),
        None => match child.wait().await {
            Ok(status) => format!("exited with {status}"),
            Err(err) => format!("could not determine exit status: {err}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

    /// Binds `route(path, spec)` on an ephemeral local port and returns
    /// its ws:// URL, running the server in a background task for the
    /// life of the test process (fine for a short-lived test binary).
    async fn spawn_server(path: &str, spec: EngineSpec) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let app = route(path, spec);
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        format!("ws://{addr}{path}")
    }

    #[tokio::test]
    async fn relays_a_real_uci_handshake() {
        // `cat` is a stand-in "engine": it echoes stdin back on stdout,
        // so sending a line and reading the same line back confirms the
        // relay's stdin-write / stdout-read plumbing works end to end
        // over a real WebSocket upgrade, without depending on a real
        // chess engine binary being present in this test environment.
        let spec = EngineSpec {
            argv: vec!["cat".to_string()],
            cwd: std::env::temp_dir(),
        };
        let url = spawn_server("/ws/echo", spec).await;

        let (mut ws, _) = tokio_tungstenite::connect_async(&url)
            .await
            .expect("connect");
        ws.send(ClientMessage::text("hello uci"))
            .await
            .expect("send");

        let reply = ws.next().await.expect("a reply").expect("not an error");
        assert_eq!(reply.into_text().unwrap(), "hello uci");
    }

    #[tokio::test]
    async fn reports_and_closes_when_the_engine_process_cannot_even_start() {
        // A command that fails to spawn at all (unlike one that spawns
        // and then exits) -- exercises the "failed to spawn" early
        // return in `relay`, distinct from `exit_reason`'s path.
        let spec = EngineSpec {
            argv: vec!["/no/such/binary/exists".to_string()],
            cwd: std::env::temp_dir(),
        };
        let url = spawn_server("/ws/broken", spec).await;

        let (mut ws, _) = tokio_tungstenite::connect_async(&url)
            .await
            .expect("connect");

        // Nothing to relay -- the spawn failed, so the socket should
        // just close (possibly after other queued messages, but here
        // there are none) rather than hang forever.
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next()).await;
        assert!(
            outcome.is_ok(),
            "socket should close promptly, not hang, when the engine can't spawn at all"
        );
    }

    #[tokio::test]
    async fn reports_the_reason_and_closes_when_the_engine_process_exits_immediately() {
        // A real process that spawns successfully but immediately exits
        // with a stderr message -- exercises `watch_for_exit`'s actual
        // purpose: the browser must see a diagnostic line and a closed
        // socket, not hang waiting for a UCI reply that will never come.
        let spec = EngineSpec {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                "echo 'boom: no module named bee_training' >&2; exit 1".to_string(),
            ],
            cwd: std::env::temp_dir(),
        };
        let url = spawn_server("/ws/crashes", spec).await;

        let (mut ws, _) = tokio_tungstenite::connect_async(&url)
            .await
            .expect("connect");

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let mut lines = Vec::new();
            while let Some(Ok(msg)) = ws.next().await {
                if let ClientMessage::Text(text) = msg {
                    lines.push(text.to_string());
                }
            }
            lines
        })
        .await
        .expect("should close within the timeout, not hang");

        assert!(
            outcome
                .iter()
                .any(|line| line.contains("boom: no module named bee_training")),
            "expected a diagnostic line naming the real failure, got: {outcome:?}"
        );
    }
}
