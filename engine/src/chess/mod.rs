//! Chess-domain core: board representation, moves, and position state.
//!
//! This module defines the chess primitives (`Square`,
//! `Piece`/`PieceKind`/`Color`, `Move`, `Position` with castling rights,
//! en passant square, halfmove clock, and side to move), FEN
//! parsing/serialization (`Position::from_fen`/`Position::to_fen`),
//! make/unmake (`Position::make_move`/`Position::unmake_move`, via the
//! `Undo` record), pseudo-legal move generation
//! (`Position::generate_pseudo_legal_moves`), attack detection plus
//! legal move generation (`Position::is_square_attacked`,
//! `Position::in_check`, `Position::generate_legal_moves`), and perft
//! (`perft`, `perft_divide`) -- completing Milestone 1's chess core.
//!
//! `Position` and `Move` are the shared vocabulary that `search`, `eval`,
//! and `uci` all depend on.

mod attacks;
mod castling;
mod fen;
mod make_unmake;
mod movegen;
mod moves;
mod perft;
mod piece;
mod position;
mod square;

pub use castling::CastlingRights;
pub use fen::FenError;
pub use make_unmake::Undo;
pub use moves::{Move, MoveFlag};
pub use perft::{perft, perft_divide};
pub use piece::{Color, Piece, PieceKind};
pub use position::Position;
pub use square::Square;
