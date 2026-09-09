//! [`GameFilter`]: narrows [`crate::GameCatalog::games`].

/// A side of the board, for [`GameFilter::color`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

/// Filters for [`crate::GameCatalog::games`]. Every field left `None`/empty
/// matches everything -- `GameFilter::default()` returns every game.
///
/// Deliberately just the handful of columns the schema actually indexes
/// (source, player, played_at) plus the cheap-to-scan ones (result, color,
/// rated) -- see the crate docs' note on not over-modeling the schema yet.
#[derive(Debug, Clone, Default)]
pub struct GameFilter {
    /// Only games from this source (e.g. `"lichess"`).
    pub source: Option<String>,
    /// Only games with this player as white or black.
    pub player: Option<String>,
    /// Only games where `player` played this color. Ignored if `player` is
    /// unset.
    pub color: Option<Color>,
    /// Only games with this result (e.g. `"1-0"`).
    pub result: Option<String>,
    /// Only games played at or after this Unix-millisecond timestamp.
    pub since: Option<i64>,
    /// Only games played at or before this Unix-millisecond timestamp.
    pub until: Option<i64>,
    /// Only rated (`true`) or only casual (`false`) games.
    pub rated: Option<bool>,
}

impl GameFilter {
    /// An unfiltered [`GameFilter`] -- matches every game. Same as
    /// [`GameFilter::default`], spelled out for readability at call sites.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    #[must_use]
    pub fn player(mut self, player: impl Into<String>) -> Self {
        self.player = Some(player.into());
        self
    }

    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    #[must_use]
    pub fn result(mut self, result: impl Into<String>) -> Self {
        self.result = Some(result.into());
        self
    }

    #[must_use]
    pub fn since(mut self, since: i64) -> Self {
        self.since = Some(since);
        self
    }

    #[must_use]
    pub fn until(mut self, until: i64) -> Self {
        self.until = Some(until);
        self
    }

    #[must_use]
    pub fn rated(mut self, rated: bool) -> Self {
        self.rated = Some(rated);
        self
    }
}
