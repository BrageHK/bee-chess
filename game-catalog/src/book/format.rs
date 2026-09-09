//! The `.book` binary artifact format (v1): what `ExperienceBookBuilder`
//! writes and what `ExperienceBook` (the engine-side `OpeningBook`
//! consumer, added in a follow-up PR) reads.
//!
//! Deliberately boring, per the design this followed: a fixed header,
//! then positions sorted by key, each carrying its candidate moves
//! sorted by descending weight. No compression, no variable-length
//! cleverness beyond the counts needed to know how much to read --
//! opening lookup happens once at the root of a search, so this format
//! optimizes for "obviously correct and easy to `include_bytes!`", not
//! for size or lookup speed.
//!
//! ```text
//! HEADER
//!   magic             b"BEEBOOK" (7 bytes)
//!   format_version    u16
//!   key_version       u16
//!   entry_count       u32
//!
//! POSITION × entry_count, sorted by key ascending
//!   key               u64  (see book::key::book_position_key)
//!   candidate_count   u16
//!   MOVE × candidate_count, sorted by weight descending
//!     encoded_move      u16  (bee_chess_core::Move's own packed bits --
//!                             see this module's docs on why that's
//!                             still an explicit, versioned format field)
//!     weight            u32
//!     games             u32
//!     score_per_mille   u16
//! ```
//!
//! All multi-byte integers are little-endian.
//!
//! ## Why `Move`'s own bit encoding, not a separate `BookMoveV1`
//!
//! `bee_chess_core::Move` is already documented as "packed into 16
//! bits" specifically because move generation needs a cheap, stable
//! representation -- the same properties a book format wants. Reusing
//! it avoids a redundant translation layer for no benefit. What matters
//! for the format *contract* is that `format_version` is bumped the
//! moment `Move`'s bit layout ever changes incompatibly (a change to
//! `chess/src/moves.rs`'s `FROM_SHIFT`/`TO_SHIFT`/`FLAG_SHIFT`/
//! `MoveFlag::to_bits`, in practice) -- exactly like any other field in
//! this format. The raw `u16` is stored and read back via
//! `Move::new`/`Move::from`/`Move::to`/`Move::flag` (never persisted as
//! a `Move` value directly), so this file is the one place that
//! contract lives.

use std::io::{self, Read, Write};

use bee_chess_core::Move;

use super::key::KEY_SCHEME_VERSION;

pub const MAGIC: &[u8; 7] = b"BEEBOOK";
pub const FORMAT_VERSION: u16 = 1;

/// One position's aggregated candidate moves, ready to write. Distinct
/// from the builder's own richer `PositionExperience` (see
/// `book::builder`) -- this is exactly the runtime shape, already
/// filtered and scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookEntry {
    pub key: u64,
    /// Sorted by weight descending; see `write`.
    pub candidates: Vec<BookCandidate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BookCandidate {
    pub mv: Move,
    pub weight: u32,
    pub games: u32,
    pub score_per_mille: u16,
}

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("I/O error reading/writing book data")]
    Io(#[from] io::Error),
    #[error("not a book file: bad magic bytes")]
    BadMagic,
    #[error("unsupported book format version {found} (this build supports {supported})")]
    UnsupportedFormatVersion { found: u16, supported: u16 },
    #[error("unsupported book key scheme version {found} (this build supports {supported})")]
    UnsupportedKeyVersion { found: u16, supported: u16 },
    #[error("book entries are not sorted by key (or contain a duplicate) at index {index}")]
    NotSorted { index: usize },
}

/// Writes `entries` (must already be sorted by `key` ascending, with no
/// duplicate keys -- see `book::builder`, which is responsible for that
/// invariant) as a `.book` artifact.
pub fn write(entries: &[BookEntry], out: &mut impl Write) -> Result<(), FormatError> {
    for window in entries.windows(2) {
        if window[0].key >= window[1].key {
            return Err(FormatError::NotSorted { index: 1 });
        }
    }

    out.write_all(MAGIC)?;
    out.write_all(&FORMAT_VERSION.to_le_bytes())?;
    out.write_all(&KEY_SCHEME_VERSION.to_le_bytes())?;
    out.write_all(&(entries.len() as u32).to_le_bytes())?;

    for entry in entries {
        out.write_all(&entry.key.to_le_bytes())?;
        out.write_all(&(entry.candidates.len() as u16).to_le_bytes())?;
        for candidate in &entry.candidates {
            let encoded: u16 = move_to_bits(candidate.mv);
            out.write_all(&encoded.to_le_bytes())?;
            out.write_all(&candidate.weight.to_le_bytes())?;
            out.write_all(&candidate.games.to_le_bytes())?;
            out.write_all(&candidate.score_per_mille.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Reads back a `.book` artifact written by `write`. Validates the
/// header but does not re-verify sort order (writers are trusted to
/// have upheld the invariant `write` itself enforces) -- a runtime
/// `ExperienceBook` doing a binary search over misordered entries would
/// simply miss/mis-hit lookups, not corrupt anything, and re-scanning
/// the whole book to check is wasted work for the common case of "an
/// artifact this same codebase wrote."
pub fn read(input: &mut impl Read) -> Result<Vec<BookEntry>, FormatError> {
    let mut magic = [0u8; 7];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(FormatError::BadMagic);
    }

    let format_version = read_u16(input)?;
    if format_version != FORMAT_VERSION {
        return Err(FormatError::UnsupportedFormatVersion {
            found: format_version,
            supported: FORMAT_VERSION,
        });
    }

    let key_version = read_u16(input)?;
    if key_version != KEY_SCHEME_VERSION {
        return Err(FormatError::UnsupportedKeyVersion {
            found: key_version,
            supported: KEY_SCHEME_VERSION,
        });
    }

    let entry_count = read_u32(input)? as usize;
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let key = read_u64(input)?;
        let candidate_count = read_u16(input)? as usize;
        let mut candidates = Vec::with_capacity(candidate_count);
        for _ in 0..candidate_count {
            let encoded = read_u16(input)?;
            let weight = read_u32(input)?;
            let games = read_u32(input)?;
            let score_per_mille = read_u16(input)?;
            candidates.push(BookCandidate {
                mv: move_from_bits(encoded),
                weight,
                games,
                score_per_mille,
            });
        }
        entries.push(BookEntry { key, candidates });
    }

    Ok(entries)
}

/// The raw bits behind a `Move` -- see this module's docs on why the
/// format stores exactly these bits rather than a redundant
/// `BookMoveV1`. `Move` exposes `from()`/`to()`/`flag()`, not its raw
/// representation, so this round-trips through those rather than
/// reaching into private fields.
fn move_to_bits(mv: Move) -> u16 {
    let from = mv.from().index() as u16;
    let to = mv.to().index() as u16;
    let flag = flag_to_bits(mv.flag());
    from | (to << 6) | (flag << 12)
}

fn move_from_bits(bits: u16) -> Move {
    use bee_chess_core::Square;
    let from = Square::new((bits & 0x3F) as u8);
    let to = Square::new(((bits >> 6) & 0x3F) as u8);
    let flag = flag_from_bits((bits >> 12) & 0xF);
    Move::new(from, to, flag)
}

fn flag_to_bits(flag: bee_chess_core::MoveFlag) -> u16 {
    use bee_chess_core::MoveFlag::*;
    match flag {
        Quiet => 0,
        DoublePawnPush => 1,
        EnPassant => 2,
        CastleKingside => 3,
        CastleQueenside => 4,
        PromoteKnight => 5,
        PromoteBishop => 6,
        PromoteRook => 7,
        PromoteQueen => 8,
    }
}

fn flag_from_bits(bits: u16) -> bee_chess_core::MoveFlag {
    use bee_chess_core::MoveFlag::*;
    match bits {
        0 => Quiet,
        1 => DoublePawnPush,
        2 => EnPassant,
        3 => CastleKingside,
        4 => CastleQueenside,
        5 => PromoteKnight,
        6 => PromoteBishop,
        7 => PromoteRook,
        8 => PromoteQueen,
        _ => Quiet, // Unreachable for a file this format's own `write` produced.
    }
}

fn read_u16(input: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    input.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    input.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(input: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bee_chess_core::{MoveFlag, Square};

    fn sample_entries() -> Vec<BookEntry> {
        vec![
            BookEntry {
                key: 1,
                candidates: vec![BookCandidate {
                    mv: Move::new(
                        Square::from_file_rank(4, 1),
                        Square::from_file_rank(4, 3),
                        MoveFlag::DoublePawnPush,
                    ),
                    weight: 731,
                    games: 42,
                    score_per_mille: 598,
                }],
            },
            BookEntry {
                key: 42,
                candidates: vec![
                    BookCandidate {
                        mv: Move::new(
                            Square::from_file_rank(3, 1),
                            Square::from_file_rank(3, 3),
                            MoveFlag::DoublePawnPush,
                        ),
                        weight: 512,
                        games: 19,
                        score_per_mille: 553,
                    },
                    BookCandidate {
                        mv: Move::new(
                            Square::from_file_rank(6, 0),
                            Square::from_file_rank(5, 2),
                            MoveFlag::Quiet,
                        ),
                        weight: 100,
                        games: 2,
                        score_per_mille: 1000,
                    },
                ],
            },
        ]
    }

    #[test]
    fn writes_and_reads_back_identical_entries() {
        let entries = sample_entries();
        let mut bytes = Vec::new();
        write(&entries, &mut bytes).unwrap();

        let read_back = read(&mut &bytes[..]).unwrap();
        assert_eq!(read_back, entries);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = b"NOTABOOK".to_vec();
        assert!(matches!(read(&mut &bytes[..]), Err(FormatError::BadMagic)));
    }

    #[test]
    fn rejects_unsorted_entries() {
        let mut entries = sample_entries();
        entries.reverse();
        let mut bytes = Vec::new();
        assert!(matches!(
            write(&entries, &mut bytes),
            Err(FormatError::NotSorted { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_keys() {
        let mut entries = sample_entries();
        entries[1].key = entries[0].key;
        let mut bytes = Vec::new();
        assert!(matches!(
            write(&entries, &mut bytes),
            Err(FormatError::NotSorted { .. })
        ));
    }

    #[test]
    fn rejects_a_future_format_version() {
        let entries = sample_entries();
        let mut bytes = Vec::new();
        write(&entries, &mut bytes).unwrap();
        bytes[7] = 0xFF; // format_version low byte, right after the 7-byte magic
        assert!(matches!(
            read(&mut &bytes[..]),
            Err(FormatError::UnsupportedFormatVersion { .. })
        ));
    }

    #[test]
    fn an_empty_book_round_trips() {
        let mut bytes = Vec::new();
        write(&[], &mut bytes).unwrap();
        assert_eq!(read(&mut &bytes[..]).unwrap(), Vec::new());
    }
}
