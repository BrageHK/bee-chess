//! Builds a deterministic `ExperienceBook` artifact (`.book` + `.json`
//! manifest) from a [`crate::GameCatalog`].
//!
//! See `builder`'s module docs for the shape of the pipeline
//! (`GameCatalog` -> replay -> aggregate W/D/L -> shrink/filter/score ->
//! sorted `BookEntry`s). The on-disk `.book` format and the position-
//! key scheme live in the separate `bee-book-format` crate (re-exported
//! here so existing call sites keep working unchanged) specifically so
//! the engine's `ExperienceBook` reader can depend on that format
//! without depending on this crate -- see `bee-book-format`'s own docs
//! and the root workspace `Cargo.toml`'s comment. Everything else in
//! this module (`builder`, `manifest`, the win-rate/shrinkage formula,
//! SAN resolution) is offline tooling the engine binary never links
//! against.

pub mod builder;
pub mod manifest;
pub mod san;

pub use bee_book_format::{book_position_key, read, write, BookCandidate, BookEntry, FormatError};
pub use builder::{build, BuildConfig, BuildReport};
pub use manifest::Manifest;
