//! `bee-game-catalog`: a persistent, SQLite-backed catalog of imported
//! chess games (Lichess, for now), plus streaming importers.
//!
//! This is offline data tooling, not part of the competition engine's
//! search hot path -- see the crate's `Cargo.toml` description and the root
//! workspace `Cargo.toml`'s comment on why it's a separate crate from
//! `engine/`.
//!
//! # Shape
//!
//! [`GameCatalog`] wraps one SQLite database: [`GameCatalog::upsert_game`]
//! is the only write path (idempotent, keyed by [`GameRecord::id`]),
//! [`GameCatalog::game`]/[`GameCatalog::games`] are the read paths, and
//! [`GameCatalog::latest_imported_at`]/[`GameCatalog::record_synced`] track
//! per-source-per-player sync watermarks so a re-run only fetches new
//! games. [`import::lichess::sync_user`] is the (currently only) importer
//! built on top of that -- it streams a user's Lichess games and upserts
//! each one as it arrives, rather than buffering the whole export.
//!
//! The schema deliberately stores one row per game (see
//! `migrations/0001_init.sql`'s docs), not one row per ply; a consumer that
//! needs per-position data (an experience-book builder, say) derives it
//! from [`GameRecord::plies`] rather than the catalog pre-exploding it.
//! That consumer -- and any Lab HTTP surface over this catalog -- is
//! intentionally out of scope for this crate; see `tools/bee-games` for the
//! CLI that exercises it today.

mod catalog;
mod error;
mod filter;
mod game;
pub mod import;

pub use catalog::GameCatalog;
pub use error::{Error, Result};
pub use filter::{Color, GameFilter};
pub use game::GameRecord;
