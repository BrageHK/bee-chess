//! Search contracts: limits, info, results, and the `Search` trait.
//!
//! These types are the shared vocabulary between the UCI adapter and the
//! search implementation. Per ADR 0001, the v1 search algorithm is
//! alpha-beta/PVS; that implementation lands in a follow-up PR
//! (`feat/search-alpha-beta`). This module only establishes the contract.
//!
//! Field names and shapes here are expected to evolve; the point of the
//! bootstrap PR is module ownership and dependency direction, not a
//! finished API.

use crate::chess::{Move, Position};

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

/// A search score, in centipawns or as a mate distance. The exact
/// normalization (mate scoring, draw handling) is defined alongside the
/// search kernel implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Score {
    Centipawns(i32),
    MateIn(i32),
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

/// A search algorithm. Implementations run to completion (bounded by
/// `limits` and/or external cancellation) and return exactly one
/// `BestMove`. Progress is expected to be reported via a side channel in
/// the real implementation (e.g. a callback or channel), which is defined
/// alongside the async UCI state machine rather than here.
pub trait Search {
    fn search(&mut self, position: &Position, limits: &SearchLimits) -> BestMove;
}
