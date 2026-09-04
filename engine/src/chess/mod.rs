//! Chess-domain core: board representation, moves, and position state.
//!
//! This module is intentionally a stub for the bootstrap PR. Move
//! generation, make/unmake, Zobrist hashing, and FEN/UCI move parsing are
//! implemented in a follow-up PR (see `feat/core-legal-moves`).
//!
//! `Position` and `Move` are the shared vocabulary that `search`, `eval`,
//! and `uci` all depend on, so their public shape is established here even
//! before the implementation lands.

/// A chess position. Placeholder for the bootstrap PR; the real
/// implementation will hold board state, side to move, castling rights,
/// en passant target, move counters, and repetition history.
#[derive(Debug, Clone, Default)]
pub struct Position;

/// A single chess move. Placeholder for the bootstrap PR; the real
/// implementation will encode from/to squares, promotion piece, and any
/// special-move flags (castling, en passant, etc.) needed for make/unmake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Move;
