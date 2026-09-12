//! [`Error`]/[`Result`] for the whole crate.

/// Errors from [`crate::GameCatalog`] and the importers.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid analysis: {0}")]
    InvalidAnalysis(String),
    #[error("database error")]
    Database(#[from] rusqlite::Error),

    #[error("database schema is at version {found}, newer than this build supports ({supported})")]
    UnsupportedSchemaVersion { found: i64, supported: i64 },

    #[error("lichess API error")]
    Lichess(#[from] litchee::LichessError),
}

pub type Result<T> = std::result::Result<T, Error>;
