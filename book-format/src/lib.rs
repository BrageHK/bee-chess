//! `bee-book-format`: the `.book` binary artifact format and position-
//! key scheme for Bee's experience books.
//!
//! This crate is the one shared contract between the offline builder
//! (`bee-game-catalog`'s `book` module, which writes `.book` files from
//! a `GameCatalog`) and the engine's runtime reader (`bee-engine`'s
//! `ExperienceBook`, an `OpeningBook` implementation). It exists
//! specifically so the engine binary never needs to depend on
//! `bee-game-catalog` -- which pulls in `rusqlite`, `litchee`, `tokio`,
//! and `reqwest` -- just to read a book someone else already built.
//! `bee-book-format` depends on nothing but `bee-chess-core` (for
//! `Move`/`Position`) and `thiserror`, so linking it into the
//! competition engine binary changes nothing about that binary's
//! dependency footprint.
//!
//! See `format`'s docs for the on-disk layout and `key`'s docs for why
//! the position-identity hash is deliberately independent of
//! `bee_chess_core::Position::zobrist_hash`.

pub mod format;
pub mod key;

pub use format::{read, write, BookCandidate, BookEntry, FormatError, FORMAT_VERSION};
pub use key::{book_position_key, KEY_SCHEME_VERSION};
