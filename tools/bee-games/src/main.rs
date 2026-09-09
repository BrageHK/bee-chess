//! `bee-games`: a thin CLI over `bee-game-catalog`, so the catalog's real
//! API is `GameCatalog` itself (see its crate docs) and not anything
//! defined here. Deliberately boring -- hand-rolled argument parsing (not
//! worth a `clap` dependency for this few flags):
//!
//!   bee-games sync lichess <username>
//!   bee-games count
//!   bee-games list --limit 20
//!   bee-games book build-experience --player <name> [--player <name> ...]
//!       [--max-ply 20] [--min-games 5] --output <path.book>
//!
//! The database path defaults to `data/games/catalog.sqlite3` under the
//! repo root (resolved the same way `lab/src/main.rs` resolves its own
//! data dir), overridable with `BEE_GAMES_DB`.

use std::path::{Path, PathBuf};

use bee_game_catalog::book::{self, BuildConfig};
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
        [cmd, sub, rest @ ..] if cmd == "book" && sub == "build-experience" => {
            build_experience(&catalog, rest)
        }
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

/// Builds an `ExperienceBook` artifact from one or more players' games
/// (pooled into one learning identity -- see `book::build`'s docs; pass
/// `--player` more than once for e.g. Bee's games played under
/// multiple Lichess accounts) and writes both the `.book` binary and
/// its `.book.json` manifest (same path, with `.json` appended). At
/// least one `--player` and `--output` are required; `run` twice with
/// the same catalog/flags is expected to produce a byte-identical
/// `.book` (see `book::builder`'s determinism tests).
fn build_experience(catalog: &GameCatalog, args: &[String]) -> Result<(), String> {
    let mut players: Vec<String> = Vec::new();
    let mut output: Option<PathBuf> = None;
    let mut config = BuildConfig::default();

    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let mut value = || {
            iter.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--player" => players.push(value()?.clone()),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--max-ply" => {
                config.max_ply = value()?
                    .parse()
                    .map_err(|_| "invalid --max-ply value".to_string())?;
            }
            "--min-games" => {
                config.min_games = value()?
                    .parse()
                    .map_err(|_| "invalid --min-games value".to_string())?;
            }
            other => return Err(format!("unrecognized flag: {other}")),
        }
    }

    if players.is_empty() {
        return Err("at least one --player is required".to_string());
    }
    let output = output.ok_or("--output is required")?;
    let player_refs: Vec<&str> = players.iter().map(String::as_str).collect();

    let (entries, report) =
        book::build(catalog, &player_refs, &config).map_err(|e| e.to_string())?;

    let mut book_bytes = Vec::new();
    book::write(&entries, &mut book_bytes).map_err(|e| e.to_string())?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&output, &book_bytes)
        .map_err(|e| format!("writing {}: {e}", output.display()))?;

    let manifest = book::Manifest::new(
        &player_refs,
        &config,
        &report,
        &book_bytes,
        builder_commit(),
    );
    let manifest_path = manifest_path_for(&output);
    std::fs::write(&manifest_path, manifest.to_json())
        .map_err(|e| format!("writing {}: {e}", manifest_path.display()))?;

    println!(
        "built {} ({} positions) from {} game(s) for {} ({} skipped as unresolvable)",
        output.display(),
        report.positions,
        report.games_considered,
        players.join(", "),
        report.games_skipped_unresolvable,
    );
    println!("manifest: {}", manifest_path.display());

    Ok(())
}

/// The manifest path for a given `.book` output path: the same path
/// with its extension replaced by `.json` (e.g. `experience-v1.book`
/// -> `experience-v1.json`), matching the `books/experience-v1.book` +
/// `books/experience-v1.json` pairing this was designed around.
fn manifest_path_for(book_path: &Path) -> PathBuf {
    book_path.with_extension("json")
}

/// The full commit hash `HEAD` was at when this book was built, for the
/// manifest's `builder_commit` field -- best-effort: `None` (rendered
/// as JSON `null`, never a fabricated placeholder) if `git` isn't
/// available or this isn't a git checkout at all, since a manifest must
/// never claim a provenance it couldn't actually determine.
fn builder_commit() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    Some(commit.trim().to_string())
}

fn print_usage() {
    eprintln!(
        "usage:\n  bee-games sync lichess <username>\n  bee-games count\n  bee-games list [--limit N]\n  bee-games book build-experience --player <name> [--player <name> ...] [--max-ply 20] [--min-games 5] --output <path.book>"
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
