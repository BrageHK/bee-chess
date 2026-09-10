//! Bee Lab: static frontend hosting plus an authoritative game-state
//! HTTP+WebSocket API under `/api/games`/`/ws/games` (see `api`/`game`
//! -- #67/#69). `POST /api/games` (optionally naming `white`/`black`
//! engines by name to drive them automatically, see
//! `api::CreateGameRequest`), `GET /api/games/:id`, `POST
//! /api/games/:id/moves` (still the way a human move, or a game with
//! no engine side at all, reaches the server), and `GET
//! /ws/games/:id` (live UCI traffic + snapshot updates). The frontend
//! (`frontend/src/labClient.ts`, `Game.tsx`) uses exactly this API and
//! never talks to an engine process directly -- it doesn't own
//! position/clocks/legality/result itself either (#69). The permissive
//! `CorsLayer` below exists specifically for that: `npm run dev`'s
//! Vite server (`:5173`) and this server (`:8080`, by default) are
//! different origins, and only plain HTTP fetches need CORS at all
//! (WebSocket connections never did).
//!
//! Serves everything from one process on one port, replacing
//! `bridge/server.py`'s three separate ports plus Vite's own dev
//! server port for Stockfish/Bee (see #68/#67a). There used to also be
//! a raw per-engine WebSocket relay here (`/ws/stockfish`, `/ws/bee`,
//! `uci_relay.rs`), mirroring the Python bridge's dumb-relay ports --
//! removed once the frontend stopped using it at all (#89): every game
//! now goes through the authoritative API above, and its
//! `GET /ws/games/:id` stream already carries the same raw UCI
//! visibility (`GameEvent::Uci`, see #80) that the old relay routes
//! existed for.
//!
//! Run as (from the repo root, after `npm --prefix frontend run build`):
//!   cargo run -p bee-lab
//!
//! Bee-Mamba-MCTS (`mamba/mamba_mcts_batched_uci`, a standalone
//! UCI binary with the NN inference native in Rust via `tch`) is
//! registered as `"bee-mamba"` below, same as Stockfish/Bee -- unlike
//! the old Python bridge's searchless variant, this one needs no
//! separate Python process. Optional, not required at startup: most
//! clones won't have `models/chess_mamba_dynamic.pt` built (see that
//! crate's README), so it's simply left out of the registry with a
//! warning rather than refusing to start the whole server.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

mod api;
mod experiment;
mod game;
mod persistence;
mod uci_process;

use game::{EngineSpec, GameStore};

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
        .insert("stockfish", stockfish_spec)
        .insert("bee", bee_spec);

    let mamba_uci_dir = root.join("mamba/mamba_mcts_batched_uci");
    let mamba_run_script = mamba_uci_dir.join("run_uci.sh");
    if mamba_run_script.exists() {
        registry.insert(
            "bee-mamba",
            EngineSpec {
                argv: vec![mamba_run_script.to_string_lossy().into_owned()],
                cwd: mamba_uci_dir,
            },
        );
    } else {
        eprintln!(
            "bee-mamba unavailable, skipping (missing {}; see mamba/mamba_mcts_batched_uci/README.md)",
            mamba_run_script.display()
        );
    }

    // Finished/aborted games survive a restart (#123) -- see
    // `persistence`'s module docs. Kept alongside the engine binaries
    // and frontend build rather than under `lab/`'s own source tree, so
    // it isn't mistaken for something `cargo clean`/version control
    // should touch.
    let games_data_dir = root.join("data/lab");
    let app = api::router(GameStore::with_data_dir(games_data_dir), registry)
        .fallback_service(ServeDir::new(&frontend_dist))
        // See the module docs above for why -- permissive (`Any`)
        // since this is a development/orchestration server (#67), not
        // something meant to be exposed to the open internet with real
        // access control to protect. Revisit if that ever changes.
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|err| panic!("failed to bind {addr}: {err}"));

    println!("bee-lab listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

/// Stockfish and Bee are mandatory: without them there's no engine for
/// `POST /api/games` to actually drive, so refuse to start -- same
/// reasoning as `bridge/server.py`'s `require`.
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
