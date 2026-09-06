//! Search contracts: limits, info, results, and the `Search` trait.
//!
//! These types are the shared vocabulary between the UCI adapter and the
//! search implementation. Per ADR 0001, the v1 search algorithm is
//! alpha-beta/PVS. The implementation includes time-bounded iterative
//! deepening, quiescence, move ordering, a transposition table, and rule-draw
//! handling; see `search::alpha_beta` and issue #6.

use crate::chess::{Move, Position};

mod alpha_beta;
mod deadline;

pub use alpha_beta::{
    search, search_iterative, search_iterative_with_history, search_iterative_with_options,
    search_with_history, search_with_options,
};

/// Toggles for experimental search features, exposed to UCI as
/// `setoption`s (see `EngineOptions` in `crate::engine`) so Bee Lab can
/// A/B one feature at a time without any frontend/Lab code needing to
/// know what the feature is -- see the design-system milestone's
/// engine-option-discovery plan. Every field defaults to `true`
/// (search's normal, strongest configuration); turning one off is
/// always a deliberate experiment, never the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchOptions {
    /// Whether to probe/store the transposition table. Disabling this
    /// does not change search *correctness*, only its speed/ordering
    /// quality -- useful for measuring the TT's actual strength
    /// contribution in isolation.
    pub use_tt: bool,
    /// Whether `depth == 0` drops into quiescence (captures/promotions/
    /// evasions until the position is quiet) or evaluates the position
    /// directly. Disabling this reintroduces the horizon effect
    /// quiescence exists to fix -- see `search::alpha_beta`'s module
    /// docs -- so it's an experiment, not something to ship disabled.
    pub use_quiescence: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            use_tt: true,
            use_quiescence: true,
        }
    }
}

/// A search score in centipawns, always from the perspective of the
/// side to move at the point the score was produced (i.e. what
/// negamax works with throughout). Plain `i32` rather than an enum:
/// negamax's `-score` negation and ply-adjusted mate arithmetic
/// (`-SCORE_MATE + ply`) both need to compose as ordinary integer
/// arithmetic, which an enum wrapping centipawns and mate distances
/// separately would only complicate.
///
/// Mate scores are encoded in the same space, offset far enough from
/// ordinary evaluations (see `SCORE_MATE`) that they never collide:
/// a score `s` with `SCORE_MATE - MAX_PLY <= |s| <= SCORE_MATE`
/// represents a forced mate, `SCORE_MATE - s` (or `SCORE_MATE + s` if
/// `s` is negative -- see `mate_in_plies`) plies away.
pub type Score = i32;

/// Larger than any real evaluation or mate score; used as the initial
/// alpha-beta window bound.
pub const SCORE_INF: Score = 32_000;

/// The score for "checkmate delivered right now" (ply 0). An actual
/// mate found N plies from the root is reported as `SCORE_MATE - N`
/// (for the winning side) so that a shorter forced mate always scores
/// higher than a longer one -- search prefers `mate in 2` over `mate
/// in 6`, and delays being mated for as long as possible when losing.
pub const SCORE_MATE: Score = 30_000;

/// If `score` is within mate range (see `SCORE_MATE`'s docs), returns
/// the number of plies to mate: positive if the side the score is
/// reported for is delivering it, negative if they're being mated.
/// Returns `None` for an ordinary (non-mate) evaluation.
#[must_use]
pub fn mate_in_plies(score: Score) -> Option<i32> {
    const MATE_THRESHOLD: Score = SCORE_MATE - 1000;
    if score >= MATE_THRESHOLD {
        Some(SCORE_MATE - score)
    } else if score <= -MATE_THRESHOLD {
        Some(-(SCORE_MATE + score))
    } else {
        None
    }
}

/// Search limits, mirroring the fields the UCI `go` command can carry.
/// All fields are optional/default so a `SearchLimits::default()` means
/// "no explicit limit" for that dimension.
#[derive(Debug, Clone, Default)]
pub struct SearchLimits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_increment_ms: Option<u64>,
    pub black_increment_ms: Option<u64>,
    pub moves_to_go: Option<u32>,
    pub infinite: bool,
    pub ponder: bool,
}

/// A periodic progress report emitted during search, corresponding to a
/// UCI `info` line.
#[derive(Debug, Clone, Default)]
pub struct SearchInfo {
    pub depth: u32,
    pub seldepth: u32,
    pub nodes: u64,
    pub nps: u64,
    pub score: Option<Score>,
    pub pv: Vec<Move>,
}

/// The terminal result of a search, corresponding to a UCI `bestmove` line.
#[derive(Debug, Clone, Copy)]
pub struct BestMove {
    pub best_move: Move,
    pub ponder: Option<Move>,
}

/// The outcome of a completed search to some depth: the move to play
/// (if any -- `None` for checkmate/stalemate at the root), the score
/// and principal variation of that line from the root side to move's
/// perspective, how many nodes were visited, and the depth actually
/// completed (for a fixed-depth search, always the requested depth;
/// for iterative deepening, the last depth that finished before the
/// time budget ran out). This is intentionally separate from the
/// `Search` trait below (which is the eventual UCI-facing shape, not
/// yet wired up to a real implementation) -- it gives search somewhere
/// to report diagnostics from day one without waiting for
/// cancellation/threading to exist.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: Score,
    pub nodes: u64,
    pub depth: u32,
    pub pv: Vec<Move>,
}

/// A search algorithm. Implementations run to completion (bounded by
/// `limits` and/or external cancellation) and return exactly one
/// `BestMove`. Progress is expected to be reported via a side channel in
/// the real implementation (e.g. a callback or channel), which is defined
/// alongside the async UCI state machine rather than here.
pub trait Search {
    fn search(&mut self, position: &Position, limits: &SearchLimits) -> BestMove;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mate_in_plies_recognizes_winning_mate_scores() {
        assert_eq!(mate_in_plies(SCORE_MATE), Some(0));
        assert_eq!(mate_in_plies(SCORE_MATE - 2), Some(2));
    }

    #[test]
    fn mate_in_plies_recognizes_losing_mate_scores() {
        assert_eq!(mate_in_plies(-SCORE_MATE), Some(0));
        assert_eq!(mate_in_plies(-(SCORE_MATE - 2)), Some(-2));
    }

    #[test]
    fn mate_in_plies_is_none_for_ordinary_scores() {
        assert_eq!(mate_in_plies(0), None);
        assert_eq!(mate_in_plies(900), None);
        assert_eq!(mate_in_plies(-900), None);
    }

    #[test]
    fn shorter_mate_scores_higher_than_longer_mate() {
        let mate_in_2 = SCORE_MATE - 2;
        let mate_in_6 = SCORE_MATE - 6;
        assert!(mate_in_2 > mate_in_6);
    }
}
