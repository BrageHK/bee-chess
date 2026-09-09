//! Builds a deterministic `ExperienceBook` artifact (`.book` +
//! `.book.json` manifest) from a [`crate::GameCatalog`].
//!
//! See `builder`'s module docs for the shape of the pipeline
//! (`GameCatalog` -> replay -> aggregate W/D/L -> shrink/filter/score ->
//! sorted `BookEntry`s) and `format`'s docs for the on-disk layout.
//! Nothing in this module is consumed by the engine directly -- an
//! `ExperienceBook` reader/`OpeningBook` implementation is a follow-up
//! PR that only needs `format::read` and `key::book_position_key`;
//! everything else here (`builder`, `manifest`, the win-rate/shrinkage
//! formula) is offline tooling the engine binary never links against.

pub mod builder;
pub mod format;
pub mod key;
pub mod manifest;
pub mod san;

pub use builder::{build, BuildConfig, BuildReport};
pub use format::{read, write, BookCandidate, BookEntry, FormatError};
pub use key::book_position_key;
pub use manifest::Manifest;
