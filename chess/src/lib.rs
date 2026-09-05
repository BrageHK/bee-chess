//! `bee-chess-core`: canonical chess-domain types, shared by `bee-engine`
//! and `bee-lab` (and anything else in this workspace that needs to
//! know what a legal chess position/move is).
//!
//! This crate defines the chess primitives (`Square`,
//! `Piece`/`PieceKind`/`Color`, `Move`, `Position` with castling rights,
//! en passant square, halfmove clock, and side to move), FEN
//! parsing/serialization (`Position::from_fen`/`Position::to_fen`),
//! make/unmake (`Position::make_move`/`Position::unmake_move`, via the
//! `Undo` record), pseudo-legal move generation
//! (`Position::generate_pseudo_legal_moves`), attack detection plus
//! legal move generation (`Position::is_square_attacked`,
//! `Position::in_check`, `Position::generate_legal_moves`), perft
//! (`perft`, `perft_divide`), and Zobrist hashing
//! (`Position::zobrist_hash`).
//!
//! Extracted from `bee-engine` (where it originated as its `chess`
//! module) specifically so `bee-lab` (see #67/#69) can share one
//! canonical interpretation of chess rules with the competition engine,
//! rather than validating moves against a second implementation --
//! "Bee says legal, Lab says illegal" is exactly the kind of divergence
//! this crate exists to make impossible. `bee-engine` depends on this
//! crate for `Position`/`Move`/etc.; it has no reason to know about
//! search, evaluation, or UCI, and doesn't -- see this crate's own
//! zero-dependency Cargo.toml.
//!
//! `Position` and `Move` are the shared vocabulary that `bee-engine`'s
//! `search`, `eval`, and `uci` modules all depend on, and that `bee-lab`
//! will too.

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
mod zobrist;

pub use castling::CastlingRights;
pub use fen::FenError;
pub use make_unmake::Undo;
pub use moves::{Move, MoveFlag};
pub use perft::{perft, perft_divide};
pub use piece::{Color, Piece, PieceKind};
pub use position::Position;
pub use square::Square;
