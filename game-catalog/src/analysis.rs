//! Persistent post-game move/game analysis records.
//!
//! This module only defines the data model -- what an analysis run,
//! move analysis, and game analysis rollup look like, and (via
//! `GameCatalog`'s methods in `catalog.rs`) how they're stored and
//! queried. [`crate::analyzer`] handles Stockfish and writes this
//! contract; a future Lab API/problem miner can read it back out.
//!
//! # Shape
//!
//! Every [`MoveAnalysisRecord`]/[`GameAnalysisRecord`] is scoped to the
//! [`AnalysisRun`] that produced it (see [`NewAnalysisRun`]/
//! [`AnalysisRun`]'s docs) -- results from two different analyzer
//! configurations (a different engine, node budget, or schema version)
//! never get silently mixed together, and re-analyzing under the same
//! configuration is meant to add to the same run rather than needing a
//! fresh one every time (see `GameCatalog::latest_analysis_run_with_config`, which
//! is what an incremental analyzer uses to find "the run to keep
//! appending to" instead of always starting a new one).

/// One position's-worth of move analysis, as read back from storage.
/// See [`NewMoveAnalysis`] for the write side -- this carries the
/// storage-assigned `id`/`analysis_run_id` in addition to every field a
/// caller provides when recording it.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveAnalysisRecord {
    pub id: i64,
    pub analysis_run_id: i64,
    pub game_id: String,
    /// 0-indexed ply within the game (matches `GameRecord::plies`'
    /// indexing).
    pub ply: u32,
    pub fen_before: String,
    /// UCI notation (e.g. `"e2e4"`), not `bee_chess_core::Move`'s
    /// packed bits -- see this crate's docs on why a table meant to be
    /// queried/inspected directly (SQL, a future Lab UI) favors a
    /// human-readable move notation over a compact binary one.
    pub played_move: String,
    /// The analyzer's suggested move. The offline analyzer always supplies
    /// one before a played move, including losing positions.
    pub best_move: Option<String>,
    /// Both CP scores use the mover's perspective in the offline analyzer.
    pub eval_before_cp: Option<i32>,
    pub eval_after_cp: Option<i32>,
    /// How much `played_move` lost relative to the analyzer's own best
    /// move, in centipawns, from the mover's perspective (always
    /// non-negative in the ordinary case; the analyzer is responsible
    /// for computing this consistently since it depends on how it
    /// reports/negates scores across the move boundary).
    pub centipawn_loss: Option<i32>,
    /// Signed plies to mate, positive when the mover wins (offline analysis
    /// version 2). NULL for non-mate scores. Each distance is measured from
    /// its respective position; `mate_after = 0` means delivered checkmate.
    pub mate_before: Option<i32>,
    pub mate_after: Option<i32>,
    pub phase: GamePhase,
    /// NULL for records predating the offline analyzer.
    pub mover_color: Option<crate::Color>,
    pub is_bee: Option<bool>,
    /// Space-separated UCI moves from the search before the played move.
    pub pv: Option<String>,
}

/// A [`MoveAnalysisRecord`] not yet written -- everything a caller
/// supplies; `GameCatalog::record_move_analysis` assigns `id`.
#[derive(Debug, Clone, PartialEq)]
pub struct NewMoveAnalysis {
    pub analysis_run_id: i64,
    pub game_id: String,
    pub ply: u32,
    pub fen_before: String,
    pub played_move: String,
    pub best_move: Option<String>,
    pub eval_before_cp: Option<i32>,
    pub eval_after_cp: Option<i32>,
    pub centipawn_loss: Option<i32>,
    pub mate_before: Option<i32>,
    pub mate_after: Option<i32>,
    pub phase: GamePhase,
    pub mover_color: Option<crate::Color>,
    pub is_bee: Option<bool>,
    pub pv: Option<String>,
}

/// Which phase of the game a position falls into. Deliberately just
/// three coarse buckets with no fixed ply/material boundaries baked in
/// here -- see this crate's design docs on not getting religious about
/// exact thresholds before real data exists; whatever boundary an
/// analyzer chooses, it's the analyzer's decision to make and stamp
/// onto each record, not something this storage layer enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

impl GamePhase {
    const fn as_str(self) -> &'static str {
        match self {
            GamePhase::Opening => "opening",
            GamePhase::Middlegame => "middlegame",
            GamePhase::Endgame => "endgame",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "opening" => Some(GamePhase::Opening),
            "middlegame" => Some(GamePhase::Middlegame),
            "endgame" => Some(GamePhase::Endgame),
            _ => None,
        }
    }
}

/// A game-level analysis rollup, as read back from storage. See
/// [`NewGameAnalysis`] for the write side.
#[derive(Debug, Clone, PartialEq)]
pub struct GameAnalysisRecord {
    pub analysis_run_id: i64,
    pub game_id: String,
    /// Which color Bee played in this game -- a game analysis rollup
    /// represents one side. Per-move records separately identify the mover.
    pub bee_color: crate::filter::Color,
    pub avg_centipawn_loss: Option<f64>,
    pub worst_move_cp_loss: Option<i32>,
    pub inaccuracies: u32,
    pub mistakes: u32,
    pub blunders: u32,
    pub opening_avg_loss: Option<f64>,
    pub middlegame_avg_loss: Option<f64>,
    pub endgame_avg_loss: Option<f64>,
}

/// A [`GameAnalysisRecord`] not yet written.
#[derive(Debug, Clone, PartialEq)]
pub struct NewGameAnalysis {
    pub analysis_run_id: i64,
    pub game_id: String,
    pub bee_color: crate::filter::Color,
    pub avg_centipawn_loss: Option<f64>,
    pub worst_move_cp_loss: Option<i32>,
    pub inaccuracies: u32,
    pub mistakes: u32,
    pub blunders: u32,
    pub opening_avg_loss: Option<f64>,
    pub middlegame_avg_loss: Option<f64>,
    pub endgame_avg_loss: Option<f64>,
}

/// One analysis configuration that has been run, as read back from
/// storage -- see [`NewAnalysisRun`] for the write side and this
/// module's docs for why every analysis record is scoped to one of
/// these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisRun {
    pub id: i64,
    pub engine: String,
    pub nodes_per_position: Option<i64>,
    pub multipv: Option<i64>,
    pub schema_version: i64,
    pub created_at: i64,
    /// Canonical analyzer configuration, including engine digest and Bee identities.
    pub configuration: Option<String>,
}

/// An [`AnalysisRun`] not yet recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewAnalysisRun {
    /// The analyzer's own identity, e.g. `"Stockfish 18"` -- free-form,
    /// since this storage layer has no fixed notion of which engines
    /// exist.
    pub engine: String,
    /// Fixed node budget per position, when the analyzer used one.
    /// Preferred over a fixed time budget for reproducibility across
    /// runs/machines -- see this crate's design docs -- but this
    /// module doesn't enforce that; an analyzer using `movetime`
    /// instead just leaves this `None`.
    pub nodes_per_position: Option<i64>,
    pub multipv: Option<i64>,
    /// The analysis *data* schema version this run's records follow --
    /// independent of `GameCatalog`'s own SQLite schema version
    /// (`catalog::SCHEMA_VERSION`). Bumped if the meaning of a
    /// stored field ever changes (e.g. a different centipawn-loss sign
    /// convention), so old and new runs are never silently compared as
    /// if they meant the same thing.
    pub schema_version: i64,
    /// Unix milliseconds, matching every other timestamp in this crate
    /// (see `GameRecord::played_at`/`imported_at`).
    pub created_at: i64,
    pub configuration: Option<String>,
}

pub(crate) fn phase_to_sql(phase: GamePhase) -> &'static str {
    phase.as_str()
}

pub(crate) fn phase_from_sql(value: &str) -> Option<GamePhase> {
    GamePhase::parse(value)
}
