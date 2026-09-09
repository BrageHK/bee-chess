//! [`ExperienceBook`]: an [`OpeningBook`] backed by a `.book` artifact
//! built offline from Bee's own game history (see `bee-game-catalog`'s
//! `book` module and `tools/bee-games`' `book build-experience`
//! subcommand). Unlike `CowOpeningBook`, which hardcodes a fixed setup
//! sequence, this book's moves and their weights come entirely from
//! data -- the engine binary knows nothing about SQLite, Lichess, or
//! the win-rate/shrinkage formula that produced them; it only knows how
//! to read the resulting `.book` bytes (via `bee-book-format`, this
//! crate's only new dependency for this feature) and pick the
//! highest-weight legal candidate at a given position.
//!
//! The shipped book is baked in via `include_bytes!` (see
//! `crate::engine::OpeningBookKind::book`) rather than loaded from disk
//! at runtime, so a UCI opponent or tournament harness never needs to
//! ship a book file alongside the `bee` binary.

use std::collections::HashMap;

use bee_book_format::{book_position_key, BookEntry, FormatError};

use super::{BookProbe, OpeningBook, OpeningContext};

/// A parsed `.book` artifact, ready to probe. Construct with
/// [`ExperienceBook::from_bytes`].
pub struct ExperienceBook {
    /// Indexed by position key for O(1) lookup -- a `.book` file is
    /// small (opening positions only, see the builder's `max_ply`), so
    /// there's no need for the on-disk binary-search-friendly sorted
    /// layout at runtime; a hash map is simpler and plenty fast for a
    /// lookup that happens once per `go` at the search root.
    entries: HashMap<u64, BookEntry>,
}

impl ExperienceBook {
    /// Parses a `.book` artifact from `bytes` (e.g. the output of
    /// `bee-games book build-experience`, typically reached via
    /// `include_bytes!`). Fails on bad magic, an unsupported format or
    /// key-scheme version, or truncated/malformed data -- see
    /// [`FormatError`]. Never panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        let entries = bee_book_format::read(&mut &bytes[..])?;
        Ok(Self {
            entries: entries
                .into_iter()
                .map(|entry| (entry.key, entry))
                .collect(),
        })
    }

    /// How many positions this book has an opinion about.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl OpeningBook for ExperienceBook {
    fn probe(&self, context: &OpeningContext<'_>) -> Option<BookProbe> {
        let key = book_position_key(context.position);
        let entry = self.entries.get(&key)?;

        // Candidates are written sorted by weight descending (see
        // `bee-book-format`'s format docs), so the first *legal* one is
        // the book's actual top pick -- not necessarily
        // `candidates[0]` outright, since this implementation must
        // guarantee legality itself (see `OpeningBook::probe`'s docs)
        // rather than trust the artifact blindly: a future engine
        // version's move encoding could in principle diverge from what
        // an old `.book` file assumed, even though `format_version`
        // exists specifically to prevent that in practice.
        let legal_moves = context.position.generate_legal_moves();
        let candidate = entry
            .candidates
            .iter()
            .find(|candidate| legal_moves.contains(&candidate.mv))?;

        Some(BookProbe {
            mv: candidate.mv,
            games: Some(candidate.games),
            score_per_mille: Some(candidate.score_per_mille),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Move, MoveFlag, Position, Square};
    use bee_book_format::BookCandidate;

    fn book_with(entries: Vec<BookEntry>) -> ExperienceBook {
        let mut bytes = Vec::new();
        bee_book_format::write(&entries, &mut bytes).unwrap();
        ExperienceBook::from_bytes(&bytes).unwrap()
    }

    fn e4() -> Move {
        Move::new(
            Square::from_file_rank(4, 1),
            Square::from_file_rank(4, 3),
            MoveFlag::DoublePawnPush,
        )
    }

    #[test]
    fn probes_a_known_position_and_returns_its_stats() {
        let startpos_key = book_position_key(&Position::startpos());
        let book = book_with(vec![BookEntry {
            key: startpos_key,
            candidates: vec![BookCandidate {
                mv: e4(),
                weight: 731,
                games: 37,
                score_per_mille: 581,
            }],
        }]);

        let position = Position::startpos();
        let probe = book
            .probe(&OpeningContext {
                position: &position,
                moves: &[],
            })
            .expect("should hit");

        assert_eq!(probe.mv, e4());
        assert_eq!(probe.games, Some(37));
        assert_eq!(probe.score_per_mille, Some(581));
    }

    #[test]
    fn an_unknown_position_is_a_miss() {
        let book = book_with(Vec::new());
        let position = Position::startpos();
        assert_eq!(
            book.probe(&OpeningContext {
                position: &position,
                moves: &[],
            }),
            None
        );
    }

    #[test]
    fn picks_the_highest_weight_candidate() {
        let startpos_key = book_position_key(&Position::startpos());
        let d4 = Move::new(
            Square::from_file_rank(3, 1),
            Square::from_file_rank(3, 3),
            MoveFlag::DoublePawnPush,
        );
        let book = book_with(vec![BookEntry {
            key: startpos_key,
            candidates: vec![
                BookCandidate {
                    mv: e4(),
                    weight: 800,
                    games: 40,
                    score_per_mille: 600,
                },
                BookCandidate {
                    mv: d4,
                    weight: 500,
                    games: 20,
                    score_per_mille: 550,
                },
            ],
        }]);

        let position = Position::startpos();
        let probe = book
            .probe(&OpeningContext {
                position: &position,
                moves: &[],
            })
            .unwrap();
        assert_eq!(probe.mv, e4());
    }

    #[test]
    fn skips_a_top_candidate_that_is_not_currently_legal_and_falls_back_to_the_next_one() {
        // A position where the book's "best" candidate happens not to
        // be legal right now (simulating stale/corrupt data, or a
        // future encoding mismatch) -- ExperienceBook must not trust
        // the artifact blindly and must fall through to the next
        // legal candidate instead of returning an illegal move.
        let startpos_key = book_position_key(&Position::startpos());
        let illegal = Move::new(
            Square::from_file_rank(4, 3), // e4 -- not a legal *origin* from startpos
            Square::from_file_rank(4, 4),
            MoveFlag::Quiet,
        );
        let book = book_with(vec![BookEntry {
            key: startpos_key,
            candidates: vec![
                BookCandidate {
                    mv: illegal,
                    weight: 900,
                    games: 5,
                    score_per_mille: 999,
                },
                BookCandidate {
                    mv: e4(),
                    weight: 500,
                    games: 40,
                    score_per_mille: 600,
                },
            ],
        }]);

        let position = Position::startpos();
        let probe = book
            .probe(&OpeningContext {
                position: &position,
                moves: &[],
            })
            .unwrap();
        assert_eq!(probe.mv, e4());
    }

    #[test]
    fn every_candidate_illegal_is_a_miss_not_a_panic() {
        let startpos_key = book_position_key(&Position::startpos());
        let illegal = Move::new(
            Square::from_file_rank(4, 3),
            Square::from_file_rank(4, 4),
            MoveFlag::Quiet,
        );
        let book = book_with(vec![BookEntry {
            key: startpos_key,
            candidates: vec![BookCandidate {
                mv: illegal,
                weight: 900,
                games: 5,
                score_per_mille: 999,
            }],
        }]);

        let position = Position::startpos();
        assert_eq!(
            book.probe(&OpeningContext {
                position: &position,
                moves: &[],
            }),
            None
        );
    }

    #[test]
    fn malformed_bytes_are_an_error_not_a_panic() {
        assert!(ExperienceBook::from_bytes(b"not a book").is_err());
    }

    #[test]
    fn truncated_bytes_are_an_error_not_a_panic() {
        let startpos_key = book_position_key(&Position::startpos());
        let entries = vec![BookEntry {
            key: startpos_key,
            candidates: vec![BookCandidate {
                mv: e4(),
                weight: 731,
                games: 37,
                score_per_mille: 581,
            }],
        }];
        let mut bytes = Vec::new();
        bee_book_format::write(&entries, &mut bytes).unwrap();
        bytes.truncate(bytes.len() - 3);

        assert!(ExperienceBook::from_bytes(&bytes).is_err());
    }

    #[test]
    fn an_unsupported_format_version_is_rejected() {
        let mut bytes = Vec::new();
        bee_book_format::write(&[], &mut bytes).unwrap();
        bytes[7] = 0xFF; // format_version low byte, right after the 7-byte magic
        assert!(matches!(
            ExperienceBook::from_bytes(&bytes),
            Err(FormatError::UnsupportedFormatVersion { .. })
        ));
    }

    #[test]
    fn an_empty_book_reports_zero_length() {
        let book = book_with(Vec::new());
        assert_eq!(book.len(), 0);
        assert!(book.is_empty());
    }
}
