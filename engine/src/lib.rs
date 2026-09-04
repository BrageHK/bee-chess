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

pub mod chess;
pub mod engine;
pub mod eval;
pub mod search;
pub mod uci;
