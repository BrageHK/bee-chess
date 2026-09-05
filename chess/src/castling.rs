//! Castling rights.

/// Which castling moves are still available, tracked independently of
/// whether a castling move is currently legal (that also depends on
/// attacked squares and empty squares, checked at move-generation time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
}

impl CastlingRights {
    /// All four castling rights available, as at the start of a game.
    #[must_use]
    pub const fn all() -> Self {
        CastlingRights {
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
        }
    }

    /// No castling rights available.
    #[must_use]
    pub const fn none() -> Self {
        CastlingRights {
            white_kingside: false,
            white_queenside: false,
            black_kingside: false,
            black_queenside: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_grants_every_right() {
        let rights = CastlingRights::all();
        assert!(rights.white_kingside);
        assert!(rights.white_queenside);
        assert!(rights.black_kingside);
        assert!(rights.black_queenside);
    }

    #[test]
    fn none_grants_no_rights() {
        assert_eq!(CastlingRights::none(), CastlingRights::default());
    }
}
