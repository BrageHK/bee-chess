//! [`GameCatalog`]: the SQLite-backed store itself.

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{Error, Result};
use crate::filter::{Color, GameFilter};
use crate::game::GameRecord;

/// The schema version this build knows how to read/write, tracked via
/// SQLite's built-in `PRAGMA user_version` rather than a migrations table --
/// there's exactly one schema so far (`migrations/0001_init.sql`), so a
/// single integer pragma is enough; a real migration runner is a follow-up
/// once there's a second version to migrate between.
const SCHEMA_VERSION: i64 = 1;

/// A persistent, SQLite-backed catalog of imported games.
///
/// Construct with [`GameCatalog::open`]. Cheap to hold onto: `rusqlite`'s
/// `Connection` serializes access internally, so one `GameCatalog` is fine
/// for a short-lived CLI process like `bee-games`.
#[derive(Debug)]
pub struct GameCatalog {
    conn: Connection,
}

impl GameCatalog {
    /// Opens (creating if needed) a catalog at `path`. Applies the schema if
    /// the database is new, and refuses to open a database written by a
    /// newer, incompatible schema version.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        Self::from_connection(conn)
    }

    /// Opens an in-memory catalog. Used by tests that don't need the file
    /// round-trip itself (see also `tests/` for on-disk temp-directory
    /// coverage of `open`).
    pub fn open_in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        let catalog = Self { conn };
        catalog.init_schema()?;
        Ok(catalog)
    }

    fn init_schema(&self) -> Result<()> {
        let version: i64 = self
            .conn
            .pragma_query_value(None, "user_version", |row| row.get(0))?;

        if version == 0 {
            // Fresh database: apply the (only, so far) schema and stamp it.
            self.conn
                .execute_batch(include_str!("../migrations/0001_init.sql"))?;
            self.conn
                .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        } else if version > SCHEMA_VERSION {
            return Err(Error::UnsupportedSchemaVersion {
                found: version,
                supported: SCHEMA_VERSION,
            });
        }
        // version in 1..=SCHEMA_VERSION with more than one version defined
        // would run incremental migrations here; not needed yet.

        Ok(())
    }

    /// Inserts or updates a game, keyed by [`GameRecord::id`]. Safe to call
    /// with the same game repeatedly (a re-run import overwrites the row
    /// with the freshly fetched data rather than erroring or duplicating),
    /// which is what makes [`crate::import::lichess`] idempotent.
    pub fn upsert_game(&self, game: &GameRecord) -> Result<()> {
        self.conn.execute(
            "INSERT INTO games (
                id, source, played_at, white, black, white_rating, black_rating,
                result, termination, time_control, rated, variant, moves, raw_pgn,
                imported_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
            ON CONFLICT(id) DO UPDATE SET
                source = excluded.source,
                played_at = excluded.played_at,
                white = excluded.white,
                black = excluded.black,
                white_rating = excluded.white_rating,
                black_rating = excluded.black_rating,
                result = excluded.result,
                termination = excluded.termination,
                time_control = excluded.time_control,
                rated = excluded.rated,
                variant = excluded.variant,
                moves = excluded.moves,
                raw_pgn = excluded.raw_pgn,
                imported_at = excluded.imported_at",
            params![
                game.id,
                game.source,
                game.played_at,
                game.white,
                game.black,
                game.white_rating,
                game.black_rating,
                game.result,
                game.termination,
                game.time_control,
                game.rated,
                game.variant,
                game.moves,
                game.raw_pgn,
                game.imported_at,
            ],
        )?;
        Ok(())
    }

    /// Looks up one game by id.
    pub fn game(&self, id: &str) -> Result<Option<GameRecord>> {
        self.conn
            .query_row(
                &format!("{SELECT_GAME} WHERE id = ?1"),
                params![id],
                row_to_game,
            )
            .optional()
            .map_err(Error::from)
    }

    /// Games matching `filter`, most recently played first (games with no
    /// `played_at` sort last). Loaded eagerly into a `Vec` -- this is a
    /// data-tooling crate operating on one user's game history, not an
    /// unbounded corpus, so there's no streaming-iterator API here yet; add
    /// one if a catalog grows large enough to need it.
    pub fn games(&self, filter: &GameFilter) -> Result<Vec<GameRecord>> {
        let mut sql = SELECT_GAME.to_string();
        let mut clauses = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(source) = &filter.source {
            clauses.push(format!("source = ?{}", values.len() + 1));
            values.push(Box::new(source.clone()));
        }
        if let Some(player) = &filter.player {
            match filter.color {
                Some(Color::White) => {
                    clauses.push(format!("white = ?{}", values.len() + 1));
                    values.push(Box::new(player.clone()));
                }
                Some(Color::Black) => {
                    clauses.push(format!("black = ?{}", values.len() + 1));
                    values.push(Box::new(player.clone()));
                }
                None => {
                    let i = values.len() + 1;
                    clauses.push(format!("(white = ?{i} OR black = ?{i})"));
                    values.push(Box::new(player.clone()));
                }
            }
        }
        if let Some(result) = &filter.result {
            clauses.push(format!("result = ?{}", values.len() + 1));
            values.push(Box::new(result.clone()));
        }
        if let Some(since) = filter.since {
            clauses.push(format!("played_at >= ?{}", values.len() + 1));
            values.push(Box::new(since));
        }
        if let Some(until) = filter.until {
            clauses.push(format!("played_at <= ?{}", values.len() + 1));
            values.push(Box::new(until));
        }
        if let Some(rated) = filter.rated {
            clauses.push(format!("rated = ?{}", values.len() + 1));
            values.push(Box::new(rated));
        }

        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(" ORDER BY played_at IS NULL, played_at DESC");

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = values.iter().map(|v| v.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), row_to_game)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Error::from)
    }

    /// Counts games matching `filter`, without materializing them.
    pub fn count(&self, filter: &GameFilter) -> Result<u64> {
        // Simplest correct implementation for this slice's scale (see
        // `games`'s docs); revisit with a dedicated COUNT(*) query if a
        // catalog ever gets large enough for this to matter.
        Ok(self.games(filter)?.len() as u64)
    }

    /// The newest `played_at` imported so far for `source`/`player`, used by
    /// [`crate::import::lichess::sync_user`] to only request games newer
    /// than the last sync. `None` means nothing has been imported yet for
    /// this pair.
    pub fn latest_imported_at(&self, source: &str, player: &str) -> Result<Option<i64>> {
        self.conn
            .query_row(
                "SELECT latest_played_at FROM sync_state WHERE source = ?1 AND player = ?2",
                params![source, player],
                |row| row.get(0),
            )
            .optional()
            .map_err(Error::from)
    }

    /// Records that `source`/`player` has now been synced up to
    /// `latest_played_at`. Only ever moves the watermark forward -- an
    /// out-of-order or partial re-sync that saw only older games must not
    /// regress it and cause newer games to be re-fetched needlessly, but
    /// also must never skip games because a watermark got set too high.
    pub fn record_synced(&self, source: &str, player: &str, latest_played_at: i64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO sync_state (source, player, latest_played_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(source, player) DO UPDATE SET
                latest_played_at = MAX(latest_played_at, excluded.latest_played_at)",
            params![source, player, latest_played_at],
        )?;
        Ok(())
    }
}

const SELECT_GAME: &str = "SELECT id, source, played_at, white, black, white_rating, \
    black_rating, result, termination, time_control, rated, variant, moves, raw_pgn, \
    imported_at FROM games";

fn row_to_game(row: &rusqlite::Row<'_>) -> rusqlite::Result<GameRecord> {
    Ok(GameRecord {
        id: row.get(0)?,
        source: row.get(1)?,
        played_at: row.get(2)?,
        white: row.get(3)?,
        black: row.get(4)?,
        white_rating: row.get(5)?,
        black_rating: row.get(6)?,
        result: row.get(7)?,
        termination: row.get(8)?,
        time_control: row.get(9)?,
        rated: row.get(10)?,
        variant: row.get(11)?,
        moves: row.get(12)?,
        raw_pgn: row.get(13)?,
        imported_at: row.get(14)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_game(id: &str) -> GameRecord {
        GameRecord {
            id: id.to_string(),
            source: "lichess".to_string(),
            played_at: Some(1_700_000_000_000),
            white: Some("alice".to_string()),
            black: Some("bob".to_string()),
            white_rating: Some(1500),
            black_rating: Some(1490),
            result: Some("1-0".to_string()),
            termination: Some("mate".to_string()),
            time_control: Some("300+3".to_string()),
            rated: Some(true),
            variant: Some("standard".to_string()),
            moves: Some("e4 e5 Nf3".to_string()),
            raw_pgn: None,
            imported_at: 1_700_000_100_000,
        }
    }

    #[test]
    fn a_fresh_catalog_has_no_games() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        assert_eq!(catalog.games(&GameFilter::all()).unwrap(), Vec::new());
        assert_eq!(catalog.game("nope").unwrap(), None);
    }

    #[test]
    fn upserted_game_round_trips_through_game_and_games() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        let game = sample_game("g1");

        catalog.upsert_game(&game).unwrap();

        assert_eq!(catalog.game("g1").unwrap(), Some(game.clone()));
        assert_eq!(catalog.games(&GameFilter::all()).unwrap(), vec![game]);
    }

    #[test]
    fn upserting_the_same_id_again_overwrites_rather_than_duplicating() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();

        let mut updated = sample_game("g1");
        updated.result = Some("0-1".to_string());
        catalog.upsert_game(&updated).unwrap();

        assert_eq!(catalog.games(&GameFilter::all()).unwrap().len(), 1);
        assert_eq!(
            catalog.game("g1").unwrap().unwrap().result,
            Some("0-1".to_string())
        );
    }

    #[test]
    fn games_filters_by_player_and_color() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap(); // alice (white) vs bob
        let mut g2 = sample_game("g2");
        g2.white = Some("carol".to_string());
        g2.black = Some("alice".to_string());
        catalog.upsert_game(&g2).unwrap();

        let alice_games = catalog.games(&GameFilter::all().player("alice")).unwrap();
        assert_eq!(alice_games.len(), 2);

        let alice_as_white = catalog
            .games(&GameFilter::all().player("alice").color(Color::White))
            .unwrap();
        assert_eq!(alice_as_white.len(), 1);
        assert_eq!(alice_as_white[0].id, "g1");

        let alice_as_black = catalog
            .games(&GameFilter::all().player("alice").color(Color::Black))
            .unwrap();
        assert_eq!(alice_as_black.len(), 1);
        assert_eq!(alice_as_black[0].id, "g2");
    }

    #[test]
    fn games_filters_by_since_and_until() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        let mut old = sample_game("old");
        old.played_at = Some(1000);
        let mut new = sample_game("new");
        new.played_at = Some(2000);
        catalog.upsert_game(&old).unwrap();
        catalog.upsert_game(&new).unwrap();

        assert_eq!(
            catalog.games(&GameFilter::all().since(1500)).unwrap(),
            vec![new.clone()]
        );
        assert_eq!(
            catalog.games(&GameFilter::all().until(1500)).unwrap(),
            vec![old]
        );
    }

    #[test]
    fn games_orders_most_recent_first() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        let mut g1 = sample_game("g1");
        g1.played_at = Some(1000);
        let mut g2 = sample_game("g2");
        g2.played_at = Some(2000);
        catalog.upsert_game(&g1).unwrap();
        catalog.upsert_game(&g2).unwrap();

        let games = catalog.games(&GameFilter::all()).unwrap();
        assert_eq!(
            games.iter().map(|g| g.id.as_str()).collect::<Vec<_>>(),
            vec!["g2", "g1"]
        );
    }

    #[test]
    fn count_matches_games_len() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        catalog.upsert_game(&sample_game("g2")).unwrap();
        assert_eq!(catalog.count(&GameFilter::all()).unwrap(), 2);
    }

    #[test]
    fn latest_imported_at_is_none_until_recorded() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        assert_eq!(
            catalog.latest_imported_at("lichess", "alice").unwrap(),
            None
        );

        catalog.record_synced("lichess", "alice", 1000).unwrap();
        assert_eq!(
            catalog.latest_imported_at("lichess", "alice").unwrap(),
            Some(1000)
        );
    }

    #[test]
    fn record_synced_never_moves_the_watermark_backward() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.record_synced("lichess", "alice", 2000).unwrap();
        catalog.record_synced("lichess", "alice", 1000).unwrap();

        assert_eq!(
            catalog.latest_imported_at("lichess", "alice").unwrap(),
            Some(2000)
        );
    }

    #[test]
    fn reopening_an_existing_database_keeps_its_schema_and_data() {
        let dir =
            std::env::temp_dir().join(format!("bee-game-catalog-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("catalog.sqlite3");

        {
            let catalog = GameCatalog::open(&path).unwrap();
            catalog.upsert_game(&sample_game("g1")).unwrap();
        }
        {
            let catalog = GameCatalog::open(&path).unwrap();
            assert_eq!(catalog.game("g1").unwrap().unwrap().id, "g1");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_schema_from_a_newer_build_is_refused_rather_than_silently_misread() {
        let dir =
            std::env::temp_dir().join(format!("bee-game-catalog-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("catalog.sqlite3");

        {
            let conn = Connection::open(&path).unwrap();
            conn.pragma_update(None, "user_version", SCHEMA_VERSION + 1)
                .unwrap();
        }

        let err = GameCatalog::open(&path).unwrap_err();
        assert!(matches!(err, Error::UnsupportedSchemaVersion { .. }));

        std::fs::remove_dir_all(&dir).ok();
    }
}
