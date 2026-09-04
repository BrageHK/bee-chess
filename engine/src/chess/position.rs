//! Chess position: board state plus everything needed to make/unmake
//! moves and detect draws.
//!
//! This module defines board layout, castling rights, en passant target,
//! halfmove clock, side to move, and a hardcoded starting position. FEN
//! parsing/serialization lives in `super::fen`. Pseudo-legal/legal move
//! generation, make/unmake, and Zobrist hashing land in follow-up PRs.

use super::castling::CastlingRights;
use super::piece::{Color, Piece, PieceKind};
use super::square::Square;

/// A standard chess position.
///
/// The board is a simple 8x8 array of optional pieces indexed by
/// `Square::index()`. This is intentionally not a bitboard
/// representation yet — correctness first, per the milestone plan;
/// faster representations can replace this internally without changing
/// the public shape established here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    board: [Option<Piece>; Square::COUNT],
    side_to_move: Color,
    castling_rights: CastlingRights,
    /// The square a pawn can capture on-passant this move, if any.
    en_passant_square: Option<Square>,
    /// Halfmove clock since the last capture or pawn move, for the
    /// fifty-move rule.
    halfmove_clock: u32,
    /// Full move number, incremented after Black's move, per FEN/PGN
    /// convention (starts at 1).
    fullmove_number: u32,
}

impl Position {
    /// An empty board: no pieces, White to move, no castling rights, no
    /// en passant square, clocks at their initial values. Mostly useful
    /// as a building block for constructing specific positions (e.g. in
    /// `Position::startpos`, or later in FEN parsing).
    #[must_use]
    pub const fn empty() -> Self {
        Position {
            board: [None; Square::COUNT],
            side_to_move: Color::White,
            castling_rights: CastlingRights::none(),
            en_passant_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    /// The standard chess starting position.
    #[must_use]
    pub fn startpos() -> Self {
        let mut position = Position {
            castling_rights: CastlingRights::all(),
            ..Position::empty()
        };

        let back_rank = [
            PieceKind::Rook,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Queen,
            PieceKind::King,
            PieceKind::Bishop,
            PieceKind::Knight,
            PieceKind::Rook,
        ];

        for (file, kind) in back_rank.iter().enumerate() {
            let file = file as u8;
            position.set_piece(
                Square::from_file_rank(file, 0),
                Some(Piece::new(*kind, Color::White)),
            );
            position.set_piece(
                Square::from_file_rank(file, 1),
                Some(Piece::new(PieceKind::Pawn, Color::White)),
            );
            position.set_piece(
                Square::from_file_rank(file, 6),
                Some(Piece::new(PieceKind::Pawn, Color::Black)),
            );
            position.set_piece(
                Square::from_file_rank(file, 7),
                Some(Piece::new(*kind, Color::Black)),
            );
        }

        position
    }

    #[must_use]
    pub const fn piece_at(&self, square: Square) -> Option<Piece> {
        self.board[square.index() as usize]
    }

    pub fn set_piece(&mut self, square: Square, piece: Option<Piece>) {
        self.board[square.index() as usize] = piece;
    }

    #[must_use]
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    pub const fn set_side_to_move(&mut self, color: Color) {
        self.side_to_move = color;
    }

    #[must_use]
    pub const fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    pub const fn set_castling_rights(&mut self, rights: CastlingRights) {
        self.castling_rights = rights;
    }

    #[must_use]
    pub const fn en_passant_square(&self) -> Option<Square> {
        self.en_passant_square
    }

    pub const fn set_en_passant_square(&mut self, square: Option<Square>) {
        self.en_passant_square = square;
    }

    #[must_use]
    pub const fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    pub const fn set_halfmove_clock(&mut self, halfmove_clock: u32) {
        self.halfmove_clock = halfmove_clock;
    }

    #[must_use]
    pub const fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }

    pub const fn set_fullmove_number(&mut self, fullmove_number: u32) {
        self.fullmove_number = fullmove_number;
    }
}

impl Default for Position {
    fn default() -> Self {
        Position::startpos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_has_no_pieces() {
        let position = Position::empty();
        for index in 0..64u8 {
            assert_eq!(position.piece_at(Square::new(index)), None);
        }
        assert_eq!(position.castling_rights(), CastlingRights::none());
        assert_eq!(position.en_passant_square(), None);
        assert_eq!(position.halfmove_clock(), 0);
    }

    #[test]
    fn startpos_places_white_back_rank() {
        let position = Position::startpos();
        assert_eq!(
            position.piece_at(Square::from_file_rank(0, 0)),
            Some(Piece::new(PieceKind::Rook, Color::White))
        );
        assert_eq!(
            position.piece_at(Square::from_file_rank(4, 0)),
            Some(Piece::new(PieceKind::King, Color::White))
        );
        assert_eq!(
            position.piece_at(Square::from_file_rank(3, 0)),
            Some(Piece::new(PieceKind::Queen, Color::White))
        );
    }

    #[test]
    fn startpos_places_black_back_rank() {
        let position = Position::startpos();
        assert_eq!(
            position.piece_at(Square::from_file_rank(0, 7)),
            Some(Piece::new(PieceKind::Rook, Color::Black))
        );
        assert_eq!(
            position.piece_at(Square::from_file_rank(4, 7)),
            Some(Piece::new(PieceKind::King, Color::Black))
        );
    }

    #[test]
    fn startpos_places_all_pawns() {
        let position = Position::startpos();
        for file in 0..8u8 {
            assert_eq!(
                position.piece_at(Square::from_file_rank(file, 1)),
                Some(Piece::new(PieceKind::Pawn, Color::White))
            );
            assert_eq!(
                position.piece_at(Square::from_file_rank(file, 6)),
                Some(Piece::new(PieceKind::Pawn, Color::Black))
            );
        }
    }

    #[test]
    fn startpos_has_empty_middle_ranks() {
        let position = Position::startpos();
        for rank in 2..6u8 {
            for file in 0..8u8 {
                assert_eq!(position.piece_at(Square::from_file_rank(file, rank)), None);
            }
        }
    }

    #[test]
    fn startpos_side_to_move_is_white() {
        assert_eq!(Position::startpos().side_to_move(), Color::White);
    }

    #[test]
    fn startpos_has_all_castling_rights() {
        assert_eq!(
            Position::startpos().castling_rights(),
            CastlingRights::all()
        );
    }

    #[test]
    fn startpos_has_no_en_passant_square() {
        assert_eq!(Position::startpos().en_passant_square(), None);
    }

    #[test]
    fn startpos_clocks_start_at_initial_values() {
        let position = Position::startpos();
        assert_eq!(position.halfmove_clock(), 0);
        assert_eq!(position.fullmove_number(), 1);
    }

    #[test]
    fn default_is_startpos() {
        assert_eq!(Position::default(), Position::startpos());
    }
}
