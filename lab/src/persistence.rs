//! Minimal disk persistence for **finished/aborted** games -- see #123
//! ("stdio logs have disappeared" when viewing a completed game).
//!
//! `GameStore` keeps every game in memory for the life of the process
//! (see `game`'s module docs), so `uci_log` never disappears while Lab
//! keeps running. It only ever vanished on a Lab restart: nothing was
//! written to disk, so `HashMap<GameId, Game>` -- and every finished
//! game's UCI history along with it -- was simply gone. Reproduced
//! directly: a finished game's full `uci_log` survives `GET
//! /api/games/:id` indefinitely within one process, and returns a
//! plain 404 the moment the process restarts.
//!
//! This is deliberately narrow, matching #123's actual scope rather
//! than #67's slice 5 in full:
//! - Only a game's **terminal** snapshot (`Finished`/`Aborted`) is ever
//!   written -- a running game is never persisted, and there is no
//!   attempt to resume a running game across a restart (a running
//!   game whose Lab process died has no engine process to resume
//!   anyway; that's a separate, larger problem).
//! - Storage is one JSON file per game (`GameSnapshot`'s existing
//!   `Serialize`/`Deserialize`), not a database -- there's no query
//!   need here beyond "load everything back at startup," and a
//!   database is explicitly called out in #67 as its own later slice.
//! - Best-effort: a write or read failure is logged and otherwise
//!   ignored. Losing a persisted log to a disk error is a regression
//!   back to today's behavior, not a new failure mode, so it must
//!   never take the game (or the server) down.

use std::path::{Path, PathBuf};

use crate::game::{GameId, GameSnapshot};

/// Where finished games are written, relative to the data directory
/// `GameStore` is constructed with (see `GameStore::with_data_dir`).
const GAMES_SUBDIR: &str = "games";

/// One finished/aborted game, written as `<data_dir>/games/<id>.json`.
/// Called after every mutation that might have just ended a game
/// (`GameStore::apply_move`/`record_move_time`/`abort`) -- cheap
/// enough (one small JSON file) to call unconditionally rather than
/// threading a "did this transition just happen" flag through each of
/// those call sites, and idempotent, so writing the same finished
/// game's snapshot again on a later call is harmless.
pub fn save(data_dir: &Path, snapshot: &GameSnapshot) {
    let dir = data_dir.join(GAMES_SUBDIR);
    if let Err(err) = std::fs::create_dir_all(&dir) {
        tracing::warn!(game_id = %snapshot.id, error = %err, "failed to create games data directory");
        return;
    }
    let path = game_path(&dir, snapshot.id);
    let json = match serde_json::to_vec_pretty(snapshot) {
        Ok(json) => json,
        Err(err) => {
            tracing::warn!(game_id = %snapshot.id, error = %err, "failed to serialize game for persistence");
            return;
        }
    };
    if let Err(err) = std::fs::write(&path, json) {
        tracing::warn!(game_id = %snapshot.id, path = %path.display(), error = %err, "failed to persist finished game");
    }
}

/// Loads every persisted game back, for `GameStore::new` to fold into
/// its startup state. Missing directory (nothing persisted yet, or a
/// fresh checkout) is not an error -- just an empty archive. A file
/// that fails to parse (corrupt, or written by an incompatible future
/// version) is skipped with a warning rather than refusing to start
/// the server over one bad record.
pub fn load_all(data_dir: &Path) -> Vec<GameSnapshot> {
    let dir = data_dir.join(GAMES_SUBDIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(err) => {
            tracing::warn!(path = %dir.display(), error = %err, "failed to read persisted games directory");
            return Vec::new();
        }
    };

    let mut snapshots = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read(&path)
            .map_err(|err| err.to_string())
            .and_then(|bytes| {
                serde_json::from_slice::<GameSnapshot>(&bytes).map_err(|err| err.to_string())
            }) {
            Ok(snapshot) => snapshots.push(snapshot),
            Err(err) => {
                tracing::warn!(path = %path.display(), error = %err, "failed to load persisted game, skipping");
            }
        }
    }
    snapshots
}

fn game_path(games_dir: &Path, id: GameId) -> PathBuf {
    games_dir.join(format!("{id}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{GameStatus, ParticipantInfo, TimeControl};

    fn finished_snapshot(id_seed: &str) -> GameSnapshot {
        // GameId has no public constructor outside `Game::new` --
        // round-trip through a real `Game` via `GameStore` instead of
        // reaching into private fields.
        let store = crate::game::GameStore::new();
        let created = store.create(
            ParticipantInfo::Human,
            ParticipantInfo::Human,
            TimeControl::fixed_move_time(100),
        );
        let mut snapshot = store.snapshot(created.id).expect("just created");
        snapshot.status = GameStatus::Aborted {
            reason: format!("test: {id_seed}"),
        };
        snapshot
    }

    #[test]
    fn a_saved_game_loads_back_identical() {
        let dir = tempdir();
        let snapshot = finished_snapshot("a");

        save(dir.path(), &snapshot);
        let loaded = load_all(dir.path());

        assert_eq!(loaded, vec![snapshot]);
    }

    #[test]
    fn loading_from_a_directory_that_does_not_exist_yet_is_an_empty_archive() {
        let dir = tempdir();
        let never_created = dir.path().join("does-not-exist");

        assert_eq!(load_all(&never_created), Vec::new());
    }

    #[test]
    fn a_corrupt_file_is_skipped_rather_than_failing_the_whole_load() {
        let dir = tempdir();
        let good = finished_snapshot("good");
        save(dir.path(), &good);

        let games_dir = dir.path().join(GAMES_SUBDIR);
        std::fs::write(games_dir.join("not-valid-json.json"), b"{ not json").unwrap();

        let loaded = load_all(dir.path());
        assert_eq!(loaded, vec![good]);
    }

    #[test]
    fn saving_the_same_game_twice_overwrites_rather_than_duplicating() {
        let dir = tempdir();
        let mut snapshot = finished_snapshot("dup");

        save(dir.path(), &snapshot);
        snapshot.moves.push("e2e4".to_string());
        save(dir.path(), &snapshot);

        let loaded = load_all(dir.path());
        assert_eq!(loaded, vec![snapshot]);
    }

    /// A unique-per-call temp directory, cleaned up on drop -- avoids
    /// pulling in a `tempfile` dependency just for these tests.
    fn tempdir() -> TempDir {
        let path =
            std::env::temp_dir().join(format!("bee-lab-persistence-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir(path)
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
