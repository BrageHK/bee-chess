//! FEN (Forsyth-Edwards Notation) parsing and serialization.
//!
//! A FEN string has six space-separated fields: piece placement, side to
//! move, castling availability, en passant target square, halfmove
//! clock, and fullmove number. See
//! <https://www.chessprogramming.org/Forsyth-Edwards_Notation>.

use std::fmt;

use super::castling::CastlingRights;
use super::piece::{Color, Piece, PieceKind};
use super::position::Position;
use super::square::Square;

/// An error encountered while parsing a FEN string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    /// The FEN did not have the required six space-separated fields.
    WrongFieldCount { found: usize },
    /// The piece placement field did not have exactly 8 ranks.
    WrongRankCount { found: usize },
    /// A rank's squares did not sum to exactly 8 files.
    WrongFileCountInRank { rank: usize, found: u32 },
    /// A character in the piece placement field was not a valid piece
    /// letter or digit 1-8.
    InvalidPieceChar { ch: char },
    /// The side-to-move field was not `w` or `b`.
    InvalidSideToMove { found: String },
    /// A character in the castling availability field was not one of
    /// `KQkq-`.
    InvalidCastlingChar { ch: char },
    /// The en passant field was not `-` or a valid algebraic square.
    InvalidEnPassantSquare { found: String },
    /// The halfmove clock field was not a valid non-negative integer.
    InvalidHalfmoveClock { found: String },
    /// The fullmove number field was not a valid positive integer.
    InvalidFullmoveNumber { found: String },
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenError::WrongFieldCount { found } => {
                write!(f, "expected 6 space-separated FEN fields, found {found}")
            }
            FenError::WrongRankCount { found } => {
                write!(f, "expected 8 ranks in piece placement, found {found}")
            }
            FenError::WrongFileCountInRank { rank, found } => {
                write!(f, "rank {rank} has {found} files, expected 8")
            }
            FenError::InvalidPieceChar { ch } => {
                write!(f, "invalid piece placement character '{ch}'")
            }
            FenError::InvalidSideToMove { found } => {
                write!(f, "invalid side to move '{found}', expected 'w' or 'b'")
            }
            FenError::InvalidCastlingChar { ch } => {
                write!(f, "invalid castling availability character '{ch}'")
            }
            FenError::InvalidEnPassantSquare { found } => {
                write!(f, "invalid en passant square '{found}'")
            }
            FenError::InvalidHalfmoveClock { found } => {
                write!(f, "invalid halfmove clock '{found}'")
            }
            FenError::InvalidFullmoveNumber { found } => {
                write!(f, "invalid fullmove number '{found}'")
            }
        }
    }
}

impl std::error::Error for FenError {}

impl Position {
    /// Parses a `Position` from a FEN string.
    ///
    /// ```
    /// use bee_engine::chess::Position;
    ///
    /// let position = Position::from_fen(
    ///     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    /// ).unwrap();
    /// assert_eq!(position, Position::startpos());
    /// ```
    pub fn from_fen(fen: &str) -> Result<Position, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(FenError::WrongFieldCount {
                found: fields.len(),
            });
        }
        let [placement, side_to_move, castling, en_passant, halfmove, fullmove] = fields[..] else {
            unreachable!("checked field count above");
        };

        let mut position = Position::empty();
        parse_placement(&mut position, placement)?;
        position.set_side_to_move(parse_side_to_move(side_to_move)?);
        position.set_castling_rights(parse_castling_rights(castling)?);
        position.set_en_passant_square(parse_en_passant(en_passant)?);
        position.set_halfmove_clock(parse_halfmove_clock(halfmove)?);
        position.set_fullmove_number(parse_fullmove_number(fullmove)?);

        Ok(position)
    }

    /// Serializes this position to a FEN string.
    ///
    /// ```
    /// use bee_engine::chess::Position;
    ///
    /// assert_eq!(
    ///     Position::startpos().to_fen(),
    ///     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    /// );
    /// ```
    #[must_use]
    pub fn to_fen(&self) -> String {
        let fields = [
            placement_to_fen(self),
            side_to_move_to_fen(self.side_to_move()),
            castling_rights_to_fen(self.castling_rights()),
            en_passant_to_fen(self.en_passant_square()),
            self.halfmove_clock().to_string(),
            self.fullmove_number().to_string(),
        ];
        fields.join(" ")
    }
}

fn parse_placement(position: &mut Position, placement: &str) -> Result<(), FenError> {
    let ranks: Vec<&str> = placement.split('/').collect();
    if ranks.len() != 8 {
        return Err(FenError::WrongRankCount { found: ranks.len() });
    }

    // FEN ranks are listed from rank 8 down to rank 1.
    for (rank_from_top, rank_str) in ranks.iter().enumerate() {
        let rank = 7 - rank_from_top as u8;
        let mut file = 0u32;

        for ch in rank_str.chars() {
            if let Some(empty_count) = ch.to_digit(10) {
                if !(1..=8).contains(&empty_count) {
                    return Err(FenError::InvalidPieceChar { ch });
                }
                file += empty_count;
            } else {
                let piece = char_to_piece(ch).ok_or(FenError::InvalidPieceChar { ch })?;
                if file >= 8 {
                    return Err(FenError::WrongFileCountInRank {
                        rank: rank as usize + 1,
                        found: file + 1,
                    });
                }
                position.set_piece(Square::from_file_rank(file as u8, rank), Some(piece));
                file += 1;
            }
        }

        if file != 8 {
            return Err(FenError::WrongFileCountInRank {
                rank: rank as usize + 1,
                found: file,
            });
        }
    }

    Ok(())
}

fn char_to_piece(ch: char) -> Option<Piece> {
    let color = if ch.is_ascii_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let kind = match ch.to_ascii_lowercase() {
        'p' => PieceKind::Pawn,
        'n' => PieceKind::Knight,
        'b' => PieceKind::Bishop,
        'r' => PieceKind::Rook,
        'q' => PieceKind::Queen,
        'k' => PieceKind::King,
        _ => return None,
    };
    Some(Piece::new(kind, color))
}

fn piece_to_char(piece: Piece) -> char {
    let ch = match piece.kind {
        PieceKind::Pawn => 'p',
        PieceKind::Knight => 'n',
        PieceKind::Bishop => 'b',
        PieceKind::Rook => 'r',
        PieceKind::Queen => 'q',
        PieceKind::King => 'k',
    };
    if piece.color == Color::White {
        ch.to_ascii_uppercase()
    } else {
        ch
    }
}

fn placement_to_fen(position: &Position) -> String {
    let mut ranks = Vec::with_capacity(8);
    for rank in (0..8u8).rev() {
        let mut rank_str = String::new();
        let mut empty_run = 0u32;
        for file in 0..8u8 {
            match position.piece_at(Square::from_file_rank(file, rank)) {
                Some(piece) => {
                    if empty_run > 0 {
                        rank_str.push_str(&empty_run.to_string());
                        empty_run = 0;
                    }
                    rank_str.push(piece_to_char(piece));
                }
                None => empty_run += 1,
            }
        }
        if empty_run > 0 {
            rank_str.push_str(&empty_run.to_string());
        }
        ranks.push(rank_str);
    }
    ranks.join("/")
}

fn parse_side_to_move(field: &str) -> Result<Color, FenError> {
    match field {
        "w" => Ok(Color::White),
        "b" => Ok(Color::Black),
        _ => Err(FenError::InvalidSideToMove {
            found: field.to_string(),
        }),
    }
}

fn side_to_move_to_fen(color: Color) -> String {
    match color {
        Color::White => "w".to_string(),
        Color::Black => "b".to_string(),
    }
}

fn parse_castling_rights(field: &str) -> Result<CastlingRights, FenError> {
    if field == "-" {
        return Ok(CastlingRights::none());
    }

    let mut rights = CastlingRights::none();
    for ch in field.chars() {
        match ch {
            'K' => rights.white_kingside = true,
            'Q' => rights.white_queenside = true,
            'k' => rights.black_kingside = true,
            'q' => rights.black_queenside = true,
            _ => return Err(FenError::InvalidCastlingChar { ch }),
        }
    }
    Ok(rights)
}

fn castling_rights_to_fen(rights: CastlingRights) -> String {
    let mut s = String::new();
    if rights.white_kingside {
        s.push('K');
    }
    if rights.white_queenside {
        s.push('Q');
    }
    if rights.black_kingside {
        s.push('k');
    }
    if rights.black_queenside {
        s.push('q');
    }
    if s.is_empty() {
        s.push('-');
    }
    s
}

fn parse_en_passant(field: &str) -> Result<Option<Square>, FenError> {
    if field == "-" {
        return Ok(None);
    }

    let mut chars = field.chars();
    let (Some(file_char), Some(rank_char), None) = (chars.next(), chars.next(), chars.next())
    else {
        return Err(FenError::InvalidEnPassantSquare {
            found: field.to_string(),
        });
    };

    if !('a'..='h').contains(&file_char) || !('1'..='8').contains(&rank_char) {
        return Err(FenError::InvalidEnPassantSquare {
            found: field.to_string(),
        });
    }

    let file = file_char as u8 - b'a';
    let rank = rank_char as u8 - b'1';
    Ok(Some(Square::from_file_rank(file, rank)))
}

fn en_passant_to_fen(square: Option<Square>) -> String {
    match square {
        Some(square) => square.to_string(),
        None => "-".to_string(),
    }
}

fn parse_halfmove_clock(field: &str) -> Result<u32, FenError> {
    field.parse().map_err(|_| FenError::InvalidHalfmoveClock {
        found: field.to_string(),
    })
}

fn parse_fullmove_number(field: &str) -> Result<u32, FenError> {
    let value: u32 = field.parse().map_err(|_| FenError::InvalidFullmoveNumber {
        found: field.to_string(),
    })?;
    if value == 0 {
        return Err(FenError::InvalidFullmoveNumber {
            found: field.to_string(),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn parses_starting_position() {
        let position = Position::from_fen(STARTPOS_FEN).expect("valid FEN");
        assert_eq!(position, Position::startpos());
    }

    #[test]
    fn starting_position_round_trips_through_fen() {
        assert_eq!(Position::startpos().to_fen(), STARTPOS_FEN);
    }

    #[test]
    fn parses_empty_board_fen() {
        let position = Position::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").expect("valid FEN");
        assert_eq!(position, Position::empty());
    }

    #[test]
    fn parses_side_to_move_black() {
        let position = Position::from_fen("8/8/8/8/8/8/8/8 b - - 0 1").expect("valid FEN");
        assert_eq!(position.side_to_move(), Color::Black);
    }

    #[test]
    fn parses_partial_castling_rights() {
        let position = Position::from_fen("8/8/8/8/8/8/8/8 w Kq - 0 1").expect("valid FEN");
        let rights = position.castling_rights();
        assert!(rights.white_kingside);
        assert!(!rights.white_queenside);
        assert!(!rights.black_kingside);
        assert!(rights.black_queenside);
    }

    #[test]
    fn parses_no_castling_rights() {
        let position = Position::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").expect("valid FEN");
        assert_eq!(position.castling_rights(), CastlingRights::none());
    }

    #[test]
    fn parses_en_passant_square() {
        let position =
            Position::from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
                .expect("valid FEN");
        assert_eq!(
            position.en_passant_square(),
            Some(Square::from_file_rank(4, 5))
        );
    }

    #[test]
    fn parses_halfmove_and_fullmove_counters() {
        let position = Position::from_fen("8/8/8/8/8/8/8/8 w - - 17 42").expect("valid FEN");
        assert_eq!(position.halfmove_clock(), 17);
        assert_eq!(position.fullmove_number(), 42);
    }

    #[test]
    fn parses_specific_piece_placement() {
        let position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid FEN");
        assert_eq!(
            position.piece_at(Square::from_file_rank(0, 0)),
            Some(Piece::new(PieceKind::Rook, Color::White))
        );
        assert_eq!(
            position.piece_at(Square::from_file_rank(7, 0)),
            Some(Piece::new(PieceKind::Rook, Color::White))
        );
        assert_eq!(
            position.piece_at(Square::from_file_rank(4, 0)),
            Some(Piece::new(PieceKind::King, Color::White))
        );
        assert_eq!(
            position.piece_at(Square::from_file_rank(0, 7)),
            Some(Piece::new(PieceKind::Rook, Color::Black))
        );
    }

    #[test]
    fn rejects_wrong_field_count() {
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 w - - 0"),
            Err(FenError::WrongFieldCount { found: 5 })
        );
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 w - - 0 1 extra"),
            Err(FenError::WrongFieldCount { found: 7 })
        );
    }

    #[test]
    fn rejects_wrong_rank_count() {
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8 w - - 0 1"),
            Err(FenError::WrongRankCount { found: 7 })
        );
    }

    #[test]
    fn rejects_wrong_file_count_in_rank() {
        // "9" is out of the valid 1-8 empty-square-run range, so it is
        // rejected as an invalid character rather than a file-count
        // mismatch.
        assert_eq!(
            Position::from_fen("9/8/8/8/8/8/8/8 w - - 0 1"),
            Err(FenError::InvalidPieceChar { ch: '9' })
        );
        assert_eq!(
            Position::from_fen("7/8/8/8/8/8/8/8 w - - 0 1"),
            Err(FenError::WrongFileCountInRank { rank: 8, found: 7 })
        );
        assert_eq!(
            Position::from_fen("pppppppp1/8/8/8/8/8/8/8 w - - 0 1"),
            Err(FenError::WrongFileCountInRank { rank: 8, found: 9 })
        );
    }

    #[test]
    fn rejects_invalid_piece_char() {
        assert_eq!(
            Position::from_fen("xxxxxxxx/8/8/8/8/8/8/8 w - - 0 1"),
            Err(FenError::InvalidPieceChar { ch: 'x' })
        );
    }

    #[test]
    fn rejects_invalid_side_to_move() {
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 x - - 0 1"),
            Err(FenError::InvalidSideToMove {
                found: "x".to_string()
            })
        );
    }

    #[test]
    fn rejects_invalid_castling_char() {
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 w KQkqx - 0 1"),
            Err(FenError::InvalidCastlingChar { ch: 'x' })
        );
    }

    #[test]
    fn rejects_invalid_en_passant_square() {
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 w - z9 0 1"),
            Err(FenError::InvalidEnPassantSquare {
                found: "z9".to_string()
            })
        );
    }

    #[test]
    fn rejects_invalid_halfmove_clock() {
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 w - - abc 1"),
            Err(FenError::InvalidHalfmoveClock {
                found: "abc".to_string()
            })
        );
    }

    #[test]
    fn rejects_invalid_fullmove_number() {
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 w - - 0 0"),
            Err(FenError::InvalidFullmoveNumber {
                found: "0".to_string()
            })
        );
        assert_eq!(
            Position::from_fen("8/8/8/8/8/8/8/8 w - - 0 abc"),
            Err(FenError::InvalidFullmoveNumber {
                found: "abc".to_string()
            })
        );
    }

    #[test]
    fn round_trips_a_complex_position() {
        let fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5";
        let position = Position::from_fen(fen).expect("valid FEN");
        assert_eq!(position.to_fen(), fen);
    }
}
