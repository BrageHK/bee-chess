//! Chess-domain core: board representation, moves, and position state.
//!
//! This module currently defines the chess primitives only (Milestone 1,
//! step 1): `Square`, `Piece`/`PieceKind`/`Color`, `Move`, `Position`
//! (with castling rights, en passant square, halfmove clock, and side to
//! move). FEN parsing, make/unmake, pseudo-legal/legal move generation,
//! and perft are separate follow-up steps in the same milestone.
//!
//! `Position` and `Move` are the shared vocabulary that `search`, `eval`,
//! and `uci` all depend on, so their public shape is established here
//! even before the rest of the implementation lands.

mod castling;
mod moves;
mod piece;
mod position;
mod square;

pub use castling::CastlingRights;
pub use moves::{Move, MoveFlag};
pub use piece::{Color, Piece, PieceKind};
pub use position::Position;
pub use square::Square;
