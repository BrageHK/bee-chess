//! Chess-domain core: board representation, moves, and position state.
//!
//! This module currently defines the chess primitives from Milestone 1's
//! first step (`Square`, `Piece`/`PieceKind`/`Color`, `Move`, `Position`
//! with castling rights, en passant square, halfmove clock, and side to
//! move) plus FEN parsing/serialization from the second step
//! (`Position::from_fen`/`Position::to_fen`). Make/unmake, pseudo-legal
//! and legal move generation, and perft are separate follow-up steps in
//! the same milestone.
//!
//! `Position` and `Move` are the shared vocabulary that `search`, `eval`,
//! and `uci` all depend on, so their public shape is established here
//! even before the rest of the implementation lands.

mod castling;
mod fen;
mod moves;
mod piece;
mod position;
mod square;

pub use castling::CastlingRights;
pub use fen::FenError;
pub use moves::{Move, MoveFlag};
pub use piece::{Color, Piece, PieceKind};
pub use position::Position;
pub use square::Square;
