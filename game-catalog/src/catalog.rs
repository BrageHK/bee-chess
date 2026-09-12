//! [`GameCatalog`]: the SQLite-backed store itself.

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{Error, Result};
use crate::filter::{Color, GameFilter};
use crate::game::GameRecord;

/// The schema version this build knows how to read/write, tracked via
/// SQLite's built-in `PRAGMA user_version`. Each version's migration file
/// lives in `migrations/000N_*.sql` and is listed in `MIGRATIONS` below,
/// in order -- see `init_schema` for how an existing database at an
/// older version is brought up to date one file at a time.
const SCHEMA_VERSION: i64 = 3;

/// Every migration this build knows how to apply, indexed by the schema
/// version it produces (i.e. `MIGRATIONS[0]` turns version 0 into
/// version 1). A fresh database runs all of them in order; an existing
/// database at version `v` runs only `MIGRATIONS[v..]`. Each file is
/// idempotent-by-construction in the sense that it only ever runs once
/// per database (guarded by `user_version`), so it's free to use plain
/// `CREATE TABLE` rather than `CREATE TABLE IF NOT EXISTS`.
const MIGRATIONS: &[&str] = &[
    include_str!("../migrations/0001_init.sql"),
    include_str!("../migrations/0002_analysis.sql"),
    include_str!("../migrations/0003_analyzer.sql"),
];

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

        if version > SCHEMA_VERSION {
            return Err(Error::UnsupportedSchemaVersion {
                found: version,
                supported: SCHEMA_VERSION,
            });
        }

        // Apply every migration this database hasn't seen yet, one at a
        // time, bumping `user_version` after each so a failure partway
        // through (a bug in a later migration, say) leaves the database
        // at a consistent, resumable version rather than either "not
        // even the first of several new migrations applied" or
        // "silently marked fully upgraded when it isn't."
        for (index, migration) in MIGRATIONS.iter().enumerate().skip(version as usize) {
            let tx = self.conn.unchecked_transaction()?;
            self.conn.execute_batch(migration)?;
            self.conn
                .pragma_update(None, "user_version", (index as i64) + 1)?;
            tx.commit()?;
        }

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

    /// Records a new analysis run and returns its assigned id. Always
    /// creates a fresh row -- an analyzer that wants to keep appending
    /// to an existing run (the normal incremental case) should look one
    /// up first via [`GameCatalog::latest_analysis_run`] and pass its id
    /// as `analysis_run_id` on subsequent `record_move_analysis`/
    /// `record_game_analysis` calls, rather than calling this again.
    pub fn record_analysis_run(&self, run: &crate::analysis::NewAnalysisRun) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO analysis_runs (
                engine, nodes_per_position, multipv, schema_version, created_at, configuration
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                run.engine,
                run.nodes_per_position,
                run.multipv,
                run.schema_version,
                run.created_at,
                run.configuration,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// The most recently created analysis run matching `engine`,
    /// `nodes_per_position`, `multipv`, and `schema_version` exactly --
    /// what an incremental analyzer calls first to decide "is there
    /// already a run for this configuration to keep appending to, or do
    /// I need to create one" (see [`GameCatalog::record_analysis_run`]'s
    /// docs). `None` means no run has ever been recorded under this
    /// exact configuration.
    pub fn latest_analysis_run(
        &self,
        engine: &str,
        nodes_per_position: Option<i64>,
        multipv: Option<i64>,
        schema_version: i64,
    ) -> Result<Option<crate::analysis::AnalysisRun>> {
        self.latest_analysis_run_with_config(
            engine,
            nodes_per_position,
            multipv,
            schema_version,
            None,
        )
    }

    /// Finds a run with an exact configuration match, including provenance.
    pub fn latest_analysis_run_with_config(
        &self,
        engine: &str,
        nodes_per_position: Option<i64>,
        multipv: Option<i64>,
        schema_version: i64,
        configuration: Option<&str>,
    ) -> Result<Option<crate::analysis::AnalysisRun>> {
        self.conn
            .query_row(
                "SELECT id, engine, nodes_per_position, multipv, schema_version, created_at, configuration
                 FROM analysis_runs
                 WHERE engine = ?1
                   AND nodes_per_position IS ?2
                   AND multipv IS ?3
                   AND schema_version = ?4
                   AND configuration IS ?5
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1",
                params![engine, nodes_per_position, multipv, schema_version, configuration],
                row_to_analysis_run,
            )
            .optional()
            .map_err(Error::from)
    }

    /// Records one move's analysis under `analysis.analysis_run_id`.
    pub fn record_move_analysis(&self, analysis: &crate::analysis::NewMoveAnalysis) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO move_analysis (
                analysis_run_id, game_id, ply, fen_before, played_move, best_move,
                eval_before_cp, eval_after_cp, centipawn_loss, mate_before, mate_after, phase,
                mover_color, is_bee, pv
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                analysis.analysis_run_id,
                analysis.game_id,
                analysis.ply,
                analysis.fen_before,
                analysis.played_move,
                analysis.best_move,
                analysis.eval_before_cp,
                analysis.eval_after_cp,
                analysis.centipawn_loss,
                analysis.mate_before,
                analysis.mate_after,
                crate::analysis::phase_to_sql(analysis.phase),
                analysis.mover_color.map(color_to_sql),
                analysis.is_bee,
                analysis.pv,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Every recorded move analysis for `game_id` under `analysis_run_id`,
    /// in ply order.
    pub fn move_analyses(
        &self,
        analysis_run_id: i64,
        game_id: &str,
    ) -> Result<Vec<crate::analysis::MoveAnalysisRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, analysis_run_id, game_id, ply, fen_before, played_move, best_move,
                eval_before_cp, eval_after_cp, centipawn_loss, mate_before, mate_after, phase,
                mover_color, is_bee, pv
             FROM move_analysis
             WHERE analysis_run_id = ?1 AND game_id = ?2
             ORDER BY ply ASC",
        )?;
        let rows = stmt.query_map(params![analysis_run_id, game_id], row_to_move_analysis)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Error::from)
    }

    /// Inserts or replaces `game_id`'s rollup under `analysis.analysis_run_id`
    /// -- idempotent by `(analysis_run_id, game_id)`, so re-analyzing a
    /// game under the same run overwrites its previous rollup rather
    /// than erroring or duplicating (mirrors [`GameCatalog::upsert_game`]'s
    /// idempotency for the same reason: a re-run must be safe to retry).
    pub fn upsert_game_analysis(&self, analysis: &crate::analysis::NewGameAnalysis) -> Result<()> {
        self.conn.execute(
            "INSERT INTO game_analysis (
                analysis_run_id, game_id, bee_color, avg_centipawn_loss, worst_move_cp_loss,
                inaccuracies, mistakes, blunders, opening_avg_loss, middlegame_avg_loss,
                endgame_avg_loss
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            ON CONFLICT(analysis_run_id, game_id) DO UPDATE SET
                bee_color = excluded.bee_color,
                avg_centipawn_loss = excluded.avg_centipawn_loss,
                worst_move_cp_loss = excluded.worst_move_cp_loss,
                inaccuracies = excluded.inaccuracies,
                mistakes = excluded.mistakes,
                blunders = excluded.blunders,
                opening_avg_loss = excluded.opening_avg_loss,
                middlegame_avg_loss = excluded.middlegame_avg_loss,
                endgame_avg_loss = excluded.endgame_avg_loss",
            params![
                analysis.analysis_run_id,
                analysis.game_id,
                color_to_sql(analysis.bee_color),
                analysis.avg_centipawn_loss,
                analysis.worst_move_cp_loss,
                analysis.inaccuracies,
                analysis.mistakes,
                analysis.blunders,
                analysis.opening_avg_loss,
                analysis.middlegame_avg_loss,
                analysis.endgame_avg_loss,
            ],
        )?;
        Ok(())
    }

    /// Atomically replaces every ply and the completion marker. An interruption or
    /// failed insert leaves the previous state intact; retries cannot duplicate plies.
    pub fn record_complete_game_analysis(
        &self,
        moves: &[crate::analysis::NewMoveAnalysis],
        summary: &crate::analysis::NewGameAnalysis,
    ) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        let game = self
            .game(&summary.game_id)?
            .ok_or_else(|| Error::InvalidAnalysis(format!("unknown game {}", summary.game_id)))?;
        if moves.is_empty()
            || moves.len() != game.plies().len()
            || moves.iter().enumerate().any(|(ply, m)| {
                m.ply as usize != ply
                    || m.game_id != summary.game_id
                    || m.analysis_run_id != summary.analysis_run_id
            })
        {
            return Err(Error::InvalidAnalysis(
                "expected every ply in order for one game/run".into(),
            ));
        }
        self.conn.execute(
            "DELETE FROM move_analysis WHERE analysis_run_id = ?1 AND game_id = ?2",
            params![summary.analysis_run_id, summary.game_id],
        )?;
        for m in moves {
            self.record_move_analysis(m)?;
        }
        self.upsert_game_analysis(summary)?;
        tx.commit()?;
        Ok(())
    }

    pub fn analysis_run(&self, id: i64) -> Result<Option<crate::analysis::AnalysisRun>> {
        self.conn.query_row(
            "SELECT id, engine, nodes_per_position, multipv, schema_version, created_at, configuration
             FROM analysis_runs WHERE id = ?1", [id], row_to_analysis_run,
        ).optional().map_err(Error::from)
    }

    /// Complete-game moves only, optionally restricted to Bee and/or a phase.
    /// Ordering is stable even for ties, for repeatable worst-move reports.
    pub fn analyzed_moves(
        &self,
        run_id: i64,
        bee_only: bool,
        phase: Option<crate::analysis::GamePhase>,
    ) -> Result<Vec<crate::analysis::MoveAnalysisRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.analysis_run_id, m.game_id, m.ply, m.fen_before,
                m.played_move, m.best_move, m.eval_before_cp, m.eval_after_cp,
                m.centipawn_loss, m.mate_before, m.mate_after, m.phase,
                m.mover_color, m.is_bee, m.pv
             FROM move_analysis m JOIN game_analysis g
                ON g.analysis_run_id = m.analysis_run_id AND g.game_id = m.game_id
             WHERE m.analysis_run_id = ?1 AND (?2 = 0 OR m.is_bee = 1)
                AND (?3 IS NULL OR m.phase = ?3)
             ORDER BY m.centipawn_loss DESC, m.game_id, m.ply",
        )?;
        let rows = stmt.query_map(
            params![run_id, bee_only, phase.map(crate::analysis::phase_to_sql)],
            row_to_move_analysis,
        )?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Error::from)
    }

    /// `game_id`'s rollup under `analysis_run_id`, if it's been analyzed.
    pub fn game_analysis(
        &self,
        analysis_run_id: i64,
        game_id: &str,
    ) -> Result<Option<crate::analysis::GameAnalysisRecord>> {
        self.conn
            .query_row(
                "SELECT analysis_run_id, game_id, bee_color, avg_centipawn_loss,
                    worst_move_cp_loss, inaccuracies, mistakes, blunders, opening_avg_loss,
                    middlegame_avg_loss, endgame_avg_loss
                 FROM game_analysis
                 WHERE analysis_run_id = ?1 AND game_id = ?2",
                params![analysis_run_id, game_id],
                row_to_game_analysis,
            )
            .optional()
            .map_err(Error::from)
    }

    /// The ids of every game in `candidate_game_ids` that has **not**
    /// yet been analyzed under `analysis_run_id` (checked against
    /// `game_analysis`, the game-level rollup -- a game is only
    /// considered analyzed once its rollup has been written, not merely
    /// because some of its moves have `move_analysis` rows). This is
    /// what an incremental analyzer calls to turn "every game for this
    /// player" into "just the ones still needing work" -- see this
    /// crate's design docs on why re-analyzing everything on every run
    /// would be wasteful.
    pub fn unanalyzed_game_ids(
        &self,
        analysis_run_id: i64,
        candidate_game_ids: &[String],
    ) -> Result<Vec<String>> {
        if candidate_game_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = (0..candidate_game_ids.len())
            .map(|i| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT id FROM games
             WHERE id IN ({placeholders})
               AND id NOT IN (
                   SELECT game_id FROM game_analysis WHERE analysis_run_id = ?1
               )"
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&analysis_run_id];
        params.extend(
            candidate_game_ids
                .iter()
                .map(|id| id as &dyn rusqlite::ToSql),
        );

        let rows = stmt.query_map(params.as_slice(), |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Error::from)
    }
}

fn color_to_sql(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn color_from_sql(value: &str) -> Option<Color> {
    match value {
        "white" => Some(Color::White),
        "black" => Some(Color::Black),
        _ => None,
    }
}

fn row_to_analysis_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<crate::analysis::AnalysisRun> {
    Ok(crate::analysis::AnalysisRun {
        id: row.get(0)?,
        engine: row.get(1)?,
        nodes_per_position: row.get(2)?,
        multipv: row.get(3)?,
        schema_version: row.get(4)?,
        created_at: row.get(5)?,
        configuration: row.get(6)?,
    })
}

fn row_to_move_analysis(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<crate::analysis::MoveAnalysisRecord> {
    let phase_text: String = row.get(12)?;
    let phase = crate::analysis::phase_from_sql(&phase_text).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            12,
            rusqlite::types::Type::Text,
            format!("unrecognized game phase: {phase_text}").into(),
        )
    })?;
    Ok(crate::analysis::MoveAnalysisRecord {
        id: row.get(0)?,
        analysis_run_id: row.get(1)?,
        game_id: row.get(2)?,
        ply: row.get(3)?,
        fen_before: row.get(4)?,
        played_move: row.get(5)?,
        best_move: row.get(6)?,
        eval_before_cp: row.get(7)?,
        eval_after_cp: row.get(8)?,
        centipawn_loss: row.get(9)?,
        mate_before: row.get(10)?,
        mate_after: row.get(11)?,
        phase,
        mover_color: row
            .get::<_, Option<String>>(13)?
            .map(|s| {
                color_from_sql(&s).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        13,
                        rusqlite::types::Type::Text,
                        format!("unrecognized color: {s}").into(),
                    )
                })
            })
            .transpose()?,
        is_bee: row.get(14)?,
        pv: row.get(15)?,
    })
}

fn row_to_game_analysis(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<crate::analysis::GameAnalysisRecord> {
    let color_text: String = row.get(2)?;
    let bee_color = color_from_sql(&color_text).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            2,
            rusqlite::types::Type::Text,
            format!("unrecognized color: {color_text}").into(),
        )
    })?;
    Ok(crate::analysis::GameAnalysisRecord {
        analysis_run_id: row.get(0)?,
        game_id: row.get(1)?,
        bee_color,
        avg_centipawn_loss: row.get(3)?,
        worst_move_cp_loss: row.get(4)?,
        inaccuracies: row.get(5)?,
        mistakes: row.get(6)?,
        blunders: row.get(7)?,
        opening_avg_loss: row.get(8)?,
        middlegame_avg_loss: row.get(9)?,
        endgame_avg_loss: row.get(10)?,
    })
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
    fn an_existing_v1_database_is_upgraded_to_v2_without_losing_its_data() {
        // Simulates a real upgrade: a database written by a build that
        // only knew migration 1, reopened by this build (which also
        // knows migration 2) -- `init_schema` must apply just the
        // missing migration, not re-run migration 1 (which would fail
        // outright: `CREATE TABLE games` on a table that already
        // exists) and not skip migration 2 either.
        let dir =
            std::env::temp_dir().join(format!("bee-game-catalog-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("catalog.sqlite3");

        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(MIGRATIONS[0]).unwrap();
            conn.pragma_update(None, "user_version", 1i64).unwrap();
        }

        let catalog = GameCatalog::open(&path).unwrap();
        // The v1 table and its data are untouched...
        catalog.upsert_game(&sample_game("g1")).unwrap();
        assert_eq!(catalog.game("g1").unwrap().unwrap().id, "g1");
        // ...and the v2 tables now exist and are queryable.
        let run = crate::analysis::NewAnalysisRun {
            engine: "Stockfish".to_string(),
            nodes_per_position: Some(1),
            multipv: Some(1),
            schema_version: 1,
            created_at: 0,
            configuration: None,
        };
        assert!(catalog.record_analysis_run(&run).is_ok());

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

    fn sample_run() -> crate::analysis::NewAnalysisRun {
        crate::analysis::NewAnalysisRun {
            engine: "Stockfish 18".to_string(),
            nodes_per_position: Some(250_000),
            multipv: Some(1),
            schema_version: 1,
            created_at: 1_700_000_000_000,
            configuration: None,
        }
    }

    #[test]
    fn record_analysis_run_assigns_increasing_ids() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        let first = catalog.record_analysis_run(&sample_run()).unwrap();
        let second = catalog.record_analysis_run(&sample_run()).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn latest_analysis_run_finds_an_exact_configuration_match() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        assert_eq!(
            catalog
                .latest_analysis_run("Stockfish 18", Some(250_000), Some(1), 1)
                .unwrap(),
            None
        );

        let id = catalog.record_analysis_run(&sample_run()).unwrap();
        let found = catalog
            .latest_analysis_run("Stockfish 18", Some(250_000), Some(1), 1)
            .unwrap()
            .expect("should find the run just recorded");
        assert_eq!(found.id, id);
        assert_eq!(found.engine, "Stockfish 18");

        // A different node budget is a different configuration.
        assert_eq!(
            catalog
                .latest_analysis_run("Stockfish 18", Some(500_000), Some(1), 1)
                .unwrap(),
            None
        );
    }

    #[test]
    fn latest_analysis_run_prefers_the_most_recently_created() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.record_analysis_run(&sample_run()).unwrap();
        let mut newer = sample_run();
        newer.created_at = sample_run().created_at + 1;
        let newer_id = catalog.record_analysis_run(&newer).unwrap();

        let found = catalog
            .latest_analysis_run("Stockfish 18", Some(250_000), Some(1), 1)
            .unwrap()
            .unwrap();
        assert_eq!(found.id, newer_id);
    }

    fn sample_move_analysis(
        run_id: i64,
        game_id: &str,
        ply: u32,
    ) -> crate::analysis::NewMoveAnalysis {
        crate::analysis::NewMoveAnalysis {
            analysis_run_id: run_id,
            game_id: game_id.to_string(),
            ply,
            fen_before: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
            played_move: "g2g4".to_string(),
            best_move: Some("e2e4".to_string()),
            eval_before_cp: Some(20),
            eval_after_cp: Some(-15),
            centipawn_loss: Some(35),
            mate_before: None,
            mate_after: None,
            phase: crate::analysis::GamePhase::Opening,
            mover_color: None,
            is_bee: None,
            pv: None,
        }
    }

    #[test]
    fn move_analysis_round_trips_through_record_and_read() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        let run_id = catalog.record_analysis_run(&sample_run()).unwrap();

        catalog
            .record_move_analysis(&sample_move_analysis(run_id, "g1", 0))
            .unwrap();
        catalog
            .record_move_analysis(&sample_move_analysis(run_id, "g1", 2))
            .unwrap();

        let analyses = catalog.move_analyses(run_id, "g1").unwrap();
        assert_eq!(analyses.len(), 2);
        // Ordered by ply ascending.
        assert_eq!(analyses[0].ply, 0);
        assert_eq!(analyses[1].ply, 2);
        assert_eq!(analyses[0].played_move, "g2g4");
        assert_eq!(analyses[0].best_move.as_deref(), Some("e2e4"));
        assert_eq!(analyses[0].centipawn_loss, Some(35));
        assert_eq!(analyses[0].phase, crate::analysis::GamePhase::Opening);
    }

    #[test]
    fn move_analyses_are_scoped_to_their_analysis_run() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        let run_a = catalog.record_analysis_run(&sample_run()).unwrap();
        let run_b = catalog.record_analysis_run(&sample_run()).unwrap();

        catalog
            .record_move_analysis(&sample_move_analysis(run_a, "g1", 0))
            .unwrap();

        assert_eq!(catalog.move_analyses(run_a, "g1").unwrap().len(), 1);
        assert_eq!(catalog.move_analyses(run_b, "g1").unwrap().len(), 0);
    }

    fn sample_game_analysis(run_id: i64, game_id: &str) -> crate::analysis::NewGameAnalysis {
        crate::analysis::NewGameAnalysis {
            analysis_run_id: run_id,
            game_id: game_id.to_string(),
            bee_color: Color::White,
            avg_centipawn_loss: Some(42.5),
            worst_move_cp_loss: Some(310),
            inaccuracies: 3,
            mistakes: 1,
            blunders: 1,
            opening_avg_loss: Some(10.0),
            middlegame_avg_loss: Some(70.0),
            endgame_avg_loss: Some(30.0),
        }
    }

    #[test]
    fn game_analysis_round_trips_through_upsert_and_read() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        let run_id = catalog.record_analysis_run(&sample_run()).unwrap();

        assert_eq!(catalog.game_analysis(run_id, "g1").unwrap(), None);

        catalog
            .upsert_game_analysis(&sample_game_analysis(run_id, "g1"))
            .unwrap();
        let found = catalog.game_analysis(run_id, "g1").unwrap().unwrap();
        assert_eq!(found.bee_color, Color::White);
        assert_eq!(found.blunders, 1);
        assert_eq!(found.avg_centipawn_loss, Some(42.5));
    }

    #[test]
    fn upsert_game_analysis_overwrites_rather_than_duplicating() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        let run_id = catalog.record_analysis_run(&sample_run()).unwrap();

        catalog
            .upsert_game_analysis(&sample_game_analysis(run_id, "g1"))
            .unwrap();
        let mut updated = sample_game_analysis(run_id, "g1");
        updated.blunders = 2;
        catalog.upsert_game_analysis(&updated).unwrap();

        let found = catalog.game_analysis(run_id, "g1").unwrap().unwrap();
        assert_eq!(found.blunders, 2);
    }

    #[test]
    fn unanalyzed_game_ids_returns_only_games_without_a_rollup_yet() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        catalog.upsert_game(&sample_game("g2")).unwrap();
        catalog.upsert_game(&sample_game("g3")).unwrap();
        let run_id = catalog.record_analysis_run(&sample_run()).unwrap();

        catalog
            .upsert_game_analysis(&sample_game_analysis(run_id, "g2"))
            .unwrap();

        let mut unanalyzed = catalog
            .unanalyzed_game_ids(
                run_id,
                &["g1".to_string(), "g2".to_string(), "g3".to_string()],
            )
            .unwrap();
        unanalyzed.sort();
        assert_eq!(unanalyzed, vec!["g1".to_string(), "g3".to_string()]);
    }

    #[test]
    fn unanalyzed_game_ids_is_empty_for_an_empty_candidate_list() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        let run_id = catalog.record_analysis_run(&sample_run()).unwrap();
        assert_eq!(
            catalog.unanalyzed_game_ids(run_id, &[]).unwrap(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn unanalyzed_game_ids_is_scoped_to_the_given_analysis_run() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        let run_a = catalog.record_analysis_run(&sample_run()).unwrap();
        let run_b = catalog.record_analysis_run(&sample_run()).unwrap();

        catalog
            .upsert_game_analysis(&sample_game_analysis(run_a, "g1"))
            .unwrap();

        // g1 is analyzed under run_a, but still unanalyzed under run_b.
        assert_eq!(
            catalog
                .unanalyzed_game_ids(run_a, &["g1".to_string()])
                .unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            catalog
                .unanalyzed_game_ids(run_b, &["g1".to_string()])
                .unwrap(),
            vec!["g1".to_string()]
        );
    }

    #[test]
    fn complete_game_replaces_partial_rows_and_rolls_back_failed_inserts() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&sample_game("g1")).unwrap();
        let run_id = catalog.record_analysis_run(&sample_run()).unwrap();
        let moves: Vec<_> = (0..3)
            .map(|ply| sample_move_analysis(run_id, "g1", ply))
            .collect();
        let summary = sample_game_analysis(run_id, "g1");
        catalog.record_move_analysis(&moves[0]).unwrap();
        catalog.record_move_analysis(&moves[0]).unwrap(); // Legacy partial duplicate.
        catalog
            .conn
            .execute_batch(
                "CREATE TRIGGER fail_analysis BEFORE INSERT ON move_analysis
             WHEN NEW.ply = 1 BEGIN SELECT RAISE(ABORT, 'injected failure'); END;",
            )
            .unwrap();
        assert!(catalog
            .record_complete_game_analysis(&moves, &summary)
            .is_err());
        assert_eq!(catalog.move_analyses(run_id, "g1").unwrap().len(), 2);
        assert!(catalog.game_analysis(run_id, "g1").unwrap().is_none());
        catalog
            .conn
            .execute_batch("DROP TRIGGER fail_analysis")
            .unwrap();
        catalog
            .record_complete_game_analysis(&moves, &summary)
            .unwrap();
        assert_eq!(catalog.move_analyses(run_id, "g1").unwrap().len(), 3);
        assert!(catalog.game_analysis(run_id, "g1").unwrap().is_some());
        assert!(catalog
            .record_complete_game_analysis(&moves[..2], &summary)
            .is_err());
        assert_eq!(catalog.move_analyses(run_id, "g1").unwrap().len(), 3);
    }

    #[test]
    fn changed_game_invalidates_analysis_but_identical_import_keeps_it() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        let mut game = sample_game("g1");
        catalog.upsert_game(&game).unwrap();
        let run_id = catalog.record_analysis_run(&sample_run()).unwrap();
        catalog
            .record_move_analysis(&sample_move_analysis(run_id, "g1", 0))
            .unwrap();
        catalog
            .upsert_game_analysis(&sample_game_analysis(run_id, "g1"))
            .unwrap();
        game.imported_at += 1;
        catalog.upsert_game(&game).unwrap();
        assert!(catalog.game_analysis(run_id, "g1").unwrap().is_some());
        game.moves = Some("d4 d5".into());
        catalog.upsert_game(&game).unwrap();
        assert!(catalog.game_analysis(run_id, "g1").unwrap().is_none());
        assert!(catalog.move_analyses(run_id, "g1").unwrap().is_empty());
    }

    #[test]
    fn v2_analysis_survives_migration_and_configuration_matches_exactly() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(MIGRATIONS[0]).unwrap();
        conn.execute_batch(MIGRATIONS[1]).unwrap();
        conn.pragma_update(None, "user_version", 2).unwrap();
        conn.execute_batch(
            "INSERT INTO games (id, source, imported_at) VALUES ('g1', 'test', 0);
             INSERT INTO analysis_runs (engine, schema_version, created_at) VALUES ('SF', 1, 0);
             INSERT INTO move_analysis (analysis_run_id, game_id, ply, fen_before, played_move, phase)
                 VALUES (1, 'g1', 0, 'fen', 'e2e4', 'opening');"
        ).unwrap();
        let catalog = GameCatalog::from_connection(conn).unwrap();
        assert!(catalog
            .analysis_run(1)
            .unwrap()
            .unwrap()
            .configuration
            .is_none());
        let old = catalog.move_analyses(1, "g1").unwrap();
        assert_eq!(old[0].played_move, "e2e4");
        assert_eq!((old[0].is_bee, &old[0].pv), (None, &None));
        let mut run = sample_run();
        run.configuration = Some("binary-a,players=bee".into());
        let id = catalog.record_analysis_run(&run).unwrap();
        let lookup = |config| {
            catalog
                .latest_analysis_run_with_config(
                    &run.engine,
                    run.nodes_per_position,
                    run.multipv,
                    run.schema_version,
                    config,
                )
                .unwrap()
        };
        assert_eq!(lookup(run.configuration.as_deref()).unwrap().id, id);
        assert!(lookup(Some("binary-b,players=bee")).is_none());
        assert!(lookup(Some("binary-a,players=opponent")).is_none());
        assert!(lookup(None).is_none());
    }
}
