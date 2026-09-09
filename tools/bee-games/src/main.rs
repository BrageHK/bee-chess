//! `bee-games`: a thin CLI over `bee-game-catalog`, so the catalog's real
//! API is `GameCatalog` itself (see its crate docs) and not anything
//! defined here. Deliberately boring for this first slice -- three
//! subcommands, hand-rolled argument parsing (not worth a `clap` dependency
//! for this few flags):
//!
//!   bee-games sync lichess <username>
//!   bee-games count
//!   bee-games list --limit 20
//!
//! The database path defaults to `data/games/catalog.sqlite3` under the
//! repo root (resolved the same way `lab/src/main.rs` resolves its own
//! data dir), overridable with `BEE_GAMES_DB`.

use std::path::{Path, PathBuf};

use bee_game_catalog::{import::lichess, GameCatalog, GameFilter};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(err) = run(&args).await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

async fn run(args: &[String]) -> Result<(), String> {
    let db_path = db_path();
    let catalog = GameCatalog::open(&db_path)
        .map_err(|err| format!("opening {}: {err}", db_path.display()))?;

    match args {
        [cmd, source, username] if cmd == "sync" && source == "lichess" => {
            sync_lichess(&catalog, username).await
        }
        [cmd] if cmd == "count" => {
            let count = catalog
                .count(&GameFilter::all())
                .map_err(|e| e.to_string())?;
            println!("{count}");
            Ok(())
        }
        [cmd, rest @ ..] if cmd == "list" => list(&catalog, rest),
        _ => {
            print_usage();
            Err("unrecognized command".to_string())
        }
    }
}

async fn sync_lichess(catalog: &GameCatalog, username: &str) -> Result<(), String> {
    let client = litchee::LichessClient::new();
    let imported = lichess::sync_user(catalog, &client, username)
        .await
        .map_err(|err| format!("syncing {username}: {err}"))?;
    println!("imported {imported} game(s) for {username}");
    Ok(())
}

fn list(catalog: &GameCatalog, rest: &[String]) -> Result<(), String> {
    let limit = parse_limit(rest)?;
    let games = catalog
        .games(&GameFilter::all())
        .map_err(|e| e.to_string())?;
    for game in games.iter().take(limit) {
        println!(
            "{} {} {} vs {} ({})",
            game.id,
            game.played_at
                .map(|t| t.to_string())
                .unwrap_or_else(|| "?".to_string()),
            game.white.as_deref().unwrap_or("?"),
            game.black.as_deref().unwrap_or("?"),
            game.result.as_deref().unwrap_or("*"),
        );
    }
    Ok(())
}

/// Parses `--limit N` out of `list`'s trailing args. Defaults to 20 -- named
/// in the module docs' usage line above, so it should stay in sync with it.
fn parse_limit(args: &[String]) -> Result<usize, String> {
    const DEFAULT: usize = 20;
    match args {
        [] => Ok(DEFAULT),
        [flag, value] if flag == "--limit" => value
            .parse()
            .map_err(|_| format!("invalid --limit value: {value}")),
        _ => Err("usage: bee-games list [--limit N]".to_string()),
    }
}

fn print_usage() {
    eprintln!(
        "usage:\n  bee-games sync lichess <username>\n  bee-games count\n  bee-games list [--limit N]"
    );
}

fn db_path() -> PathBuf {
    if let Ok(path) = std::env::var("BEE_GAMES_DB") {
        return PathBuf::from(path);
    }
    let dir = repo_root().join("data/games");
    std::fs::create_dir_all(&dir).ok();
    dir.join("catalog.sqlite3")
}

/// Repo root, resolved relative to this crate's own `Cargo.toml` location
/// (`tools/bee-games/`), matching `lab/src/main.rs`'s `repo_root` so
/// `cargo run -p bee-games` works the same regardless of invocation
/// directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tools/bee-games/ has a grandparent directory")
        .to_path_buf()
}
