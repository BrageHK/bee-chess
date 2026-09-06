//! Bee Chess engine library.
//!
//! This crate defines the shared vocabulary for the engine: chess-domain
//! types, the UCI process boundary, the search contract, and the evaluator
//! contract. See `docs/adr/0001-v1-engine-architecture.md` for the
//! architecture this crate is built around.
//!
//! Dependency direction:
//!
//! ```text
//!           UCI
//!            |
//!            v
//!      SearchController
//!            |
//!      +-----+------+
//!      v            v
//!    Search      Evaluator
//!      |
//!      v
//!  Chess/Core
//! ```
//!
//! No raw UCI strings may appear below the `uci` module.
//!
//! `chess` re-exports the `bee-chess-core` crate rather than defining
//! these types itself -- they moved there so `bee-lab` can share the
//! exact same `Position`/`Move`/legality/FEN/Zobrist implementation
//! instead of validating moves against a second one. Every existing
//! `use crate::chess::...` in this crate keeps working unchanged.

pub use bee_chess_core as chess;
pub mod book;
pub mod diagnostics;
pub mod engine;
pub mod eval;
pub mod search;
pub mod uci;
