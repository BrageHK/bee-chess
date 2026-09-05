//! Bee Lab: static frontend hosting + a UCI process relay over
//! WebSocket, replacing `bridge/server.py` for Stockfish and Bee (see
//! #68 / #67a). Serves everything from one process on one port,
//! instead of the Python bridge's three separate ports plus Vite's own
//! dev server port.
//!
//! Run as (from the repo root, after `npm --prefix frontend run build`):
//!   cargo run -p bee-lab
//!
//! Also serves an authoritative game-state HTTP API under `/api/games`
//! (see `api`/`game` -- #69/67b): `POST /api/games` (optionally naming
//! `white`/`black` engines by name to drive them automatically, see
//! `api::CreateGameRequest`), `GET /api/games/:id`, and
//! `POST /api/games/:id/moves` (still the way a human move, or a game
//! with no engine side at all, reaches the server). The frontend does
//! not use this yet -- it still owns position/clocks/move application
//! itself against the `/ws/*` relay above, exactly as it does against
//! the Python bridge today. Wiring the frontend to this API instead is
//! 69c.
//!
//! Bee-Mamba (the Python/PyTorch engine) is intentionally not served
//! here -- see #68's "out of scope." It stays on the old Python bridge
//! for now; its fate (ported here too, or left as a standalone process
//! this server doesn't know about, pending #66's model-integration
//! design) is a follow-up decision once this slice is stable.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use tower_http::services::ServeDir;

mod api;
mod game;
mod uci_process;
mod uci_relay;

use game::GameStore;
use uci_relay::EngineSpec;

const DEFAULT_PORT: u16 = 8080;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let root = repo_root();
    let stockfish_path = root.join("external/stockfish/src/stockfish");
    // Not engine/target/ -- engine/ is a member of the root Cargo
    // workspace (see /Cargo.toml), so Cargo shares one target/
    // directory across all workspace members, this crate included.
    let bee_path = root.join("target/release/bee");
    let frontend_dist = root.join("frontend/dist");

    require(&stockfish_path, "./scripts/build-stockfish.sh");
    require(&bee_path, "./scripts/build-bee.sh");
    require_dir(&frontend_dist, "npm --prefix frontend run build");

    let stockfish_spec = EngineSpec {
        argv: vec![stockfish_path.to_string_lossy().into_owned()],
        cwd: stockfish_path.parent().unwrap().to_path_buf(),
    };
    let bee_spec = EngineSpec {
        argv: vec![bee_path.to_string_lossy().into_owned()],
        cwd: bee_path.parent().unwrap().to_path_buf(),
    };

    // Stopgap engine registry (see `api::EngineRegistry`'s docs -- #70
    // is the real descriptor-based version) so `POST /api/games` can
    // name an engine by "stockfish"/"bee" instead of every caller
    // needing a binary path.
    let mut registry = api::EngineRegistry::new();
    registry
        .insert("stockfish", stockfish_spec.clone())
        .insert("bee", bee_spec.clone());

    let app = uci_relay::route("/ws/stockfish", stockfish_spec)
        .merge(uci_relay::route("/ws/bee", bee_spec))
        .merge(api::router(GameStore::new(), registry))
        .fallback_service(ServeDir::new(&frontend_dist));

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|err| panic!("failed to bind {addr}: {err}"));

    println!("bee-lab listening on http://{addr}  (stockfish + bee over /ws/*)");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

/// Stockfish and Bee are mandatory: without them there's nothing for
/// this server to usefully relay, so refuse to start -- same reasoning
/// as `bridge/server.py`'s `require`.
fn require(path: &Path, build_cmd: &str) {
    if !path.exists() {
        eprintln!(
            "missing engine binary: {}\nbuild it with: {build_cmd}",
            path.display()
        );
        std::process::exit(1);
    }
}

fn require_dir(path: &Path, build_cmd: &str) {
    if !path.is_dir() {
        eprintln!(
            "missing frontend build: {}\nbuild it with: {build_cmd}",
            path.display()
        );
        std::process::exit(1);
    }
}

/// Repo root, resolved relative to this crate's own `Cargo.toml`
/// location (`lab/`) rather than the process's current directory, so
/// `cargo run -p bee-lab` works the same regardless of where it's
/// invoked from.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("lab/ has a parent directory")
        .to_path_buf()
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
