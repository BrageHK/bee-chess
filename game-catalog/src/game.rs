//! [`GameRecord`]: one row of `games`, as returned by [`crate::GameCatalog`].

/// One imported game.
///
/// Deliberately flat and close to the `games` table -- see the crate docs
/// for why the schema stays one row per game rather than one row per ply.
/// `moves` is the game's move sequence (SAN, space-separated); positions are
/// derived from it by a caller that needs them (an experience-book builder,
/// say) rather than stored pre-exploded here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameRecord {
    /// The source's stable game id (e.g. the Lichess game id). Primary key.
    pub id: String,
    /// Where this game came from, e.g. `"lichess"`.
    pub source: String,
    /// When the game was played, Unix milliseconds. `None` if the source
    /// didn't report it.
    pub played_at: Option<i64>,
    /// The white player's name/id, as the source reports it.
    pub white: Option<String>,
    /// The black player's name/id, as the source reports it.
    pub black: Option<String>,
    /// White's rating at the time of the game.
    pub white_rating: Option<i64>,
    /// Black's rating at the time of the game.
    pub black_rating: Option<i64>,
    /// The result, e.g. `"1-0"`, `"0-1"`, `"1/2-1/2"`, `"*"`.
    pub result: Option<String>,
    /// How the game ended, e.g. `"mate"`, `"resign"`, `"timeout"`.
    pub termination: Option<String>,
    /// The time control, as the source reports it (e.g. `"300+3"`).
    pub time_control: Option<String>,
    /// Whether the game was rated.
    pub rated: Option<bool>,
    /// The variant, e.g. `"standard"`, `"chess960"`.
    pub variant: Option<String>,
    /// The move sequence in SAN, space-separated (e.g. `"e4 e5 Nf3"`).
    pub moves: Option<String>,
    /// The full PGN, when the importer fetched it.
    pub raw_pgn: Option<String>,
    /// When this row was written into the catalog, Unix milliseconds.
    pub imported_at: i64,
}

impl GameRecord {
    /// The move sequence as individual SAN tokens, in order. Empty if
    /// [`moves`](Self::moves) is `None` or blank.
    pub fn plies(&self) -> Vec<&str> {
        self.moves
            .as_deref()
            .map(|moves| moves.split_whitespace().collect())
            .unwrap_or_default()
    }
}
