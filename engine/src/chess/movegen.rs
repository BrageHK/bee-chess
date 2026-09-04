//! Pseudo-legal move generation.
//!
//! "Pseudo-legal" here means: the move follows the piece's normal
//! movement pattern and does not move onto a square occupied by a piece
//! of the same color. It does **not** check whether the side to move is
//! left in check afterward — that is legality, a follow-up milestone
//! step that will filter this module's output using
//! `Position::is_square_attacked`/`Position::in_check` (not yet
//! implemented). Castling moves generated here likewise only check that
//! the relevant squares are empty and the castling right is still held;
//! they do not check that the king is not currently in check or does
//! not pass through an attacked square, since that also depends on
//! attack detection.
//!
//! This is deliberately a simple, unoptimized generator over the
//! existing array-of-`Option<Piece>` board — no bitboards, no magic
//! sliding-attack tables. Perft (a follow-up step) is what will tell us
//! whether this needs to get faster before it needs to get more
//! clever.

use super::moves::{Move, MoveFlag};
use super::piece::{Color, Piece, PieceKind};
use super::position::Position;
use super::square::Square;

const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

const KING_OFFSETS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

const PROMOTION_KINDS: [MoveFlag; 4] = [
    MoveFlag::PromoteQueen,
    MoveFlag::PromoteRook,
    MoveFlag::PromoteBishop,
    MoveFlag::PromoteKnight,
];

impl Position {
    /// Generates all pseudo-legal moves for the side to move.
    ///
    /// See the module docs for exactly what "pseudo-legal" means here.
    #[must_use]
    pub fn generate_pseudo_legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        let side = self.side_to_move();

        for index in 0..Square::COUNT as u8 {
            let square = Square::new(index);
            let Some(piece) = self.piece_at(square) else {
                continue;
            };
            if piece.color != side {
                continue;
            }

            match piece.kind {
                PieceKind::Pawn => self.generate_pawn_moves(square, side, &mut moves),
                PieceKind::Knight => {
                    self.generate_offset_moves(square, side, &KNIGHT_OFFSETS, &mut moves);
                }
                PieceKind::Bishop => {
                    self.generate_sliding_moves(square, side, &BISHOP_DIRECTIONS, &mut moves);
                }
                PieceKind::Rook => {
                    self.generate_sliding_moves(square, side, &ROOK_DIRECTIONS, &mut moves);
                }
                PieceKind::Queen => {
                    self.generate_sliding_moves(square, side, &BISHOP_DIRECTIONS, &mut moves);
                    self.generate_sliding_moves(square, side, &ROOK_DIRECTIONS, &mut moves);
                }
                PieceKind::King => {
                    self.generate_offset_moves(square, side, &KING_OFFSETS, &mut moves);
                    self.generate_castling_moves(side, &mut moves);
                }
            }
        }

        moves
    }

    fn generate_offset_moves(
        &self,
        from: Square,
        side: Color,
        offsets: &[(i8, i8)],
        moves: &mut Vec<Move>,
    ) {
        for &(df, dr) in offsets {
            let Some(to) = offset_square(from, df, dr) else {
                continue;
            };
            if !self.occupied_by(to, side) {
                moves.push(Move::new(from, to, MoveFlag::Quiet));
            }
        }
    }

    fn generate_sliding_moves(
        &self,
        from: Square,
        side: Color,
        directions: &[(i8, i8)],
        moves: &mut Vec<Move>,
    ) {
        for &(df, dr) in directions {
            let mut current = from;
            while let Some(to) = offset_square(current, df, dr) {
                match self.piece_at(to) {
                    None => {
                        moves.push(Move::new(from, to, MoveFlag::Quiet));
                        current = to;
                    }
                    Some(occupant) if occupant.color != side => {
                        moves.push(Move::new(from, to, MoveFlag::Quiet));
                        break;
                    }
                    Some(_) => break,
                }
            }
        }
    }

    fn generate_pawn_moves(&self, from: Square, side: Color, moves: &mut Vec<Move>) {
        let (forward, start_rank, promotion_rank): (i8, u8, u8) = match side {
            Color::White => (1, 1, 7),
            Color::Black => (-1, 6, 0),
        };

        // Single push.
        if let Some(one_ahead) = offset_square(from, 0, forward) {
            if self.piece_at(one_ahead).is_none() {
                push_pawn_move(from, one_ahead, promotion_rank, moves);

                // Double push, only from the starting rank and only if
                // both squares ahead are empty.
                if from.rank() == start_rank {
                    if let Some(two_ahead) = offset_square(from, 0, forward * 2) {
                        if self.piece_at(two_ahead).is_none() {
                            moves.push(Move::new(from, two_ahead, MoveFlag::DoublePawnPush));
                        }
                    }
                }
            }
        }

        // Captures (including en passant), diagonally forward.
        for &df in &[-1i8, 1i8] {
            let Some(to) = offset_square(from, df, forward) else {
                continue;
            };

            if self.occupied_by(to, side.opposite()) {
                push_pawn_move(from, to, promotion_rank, moves);
            } else if self.en_passant_square() == Some(to) {
                moves.push(Move::new(from, to, MoveFlag::EnPassant));
            }
        }
    }

    fn generate_castling_moves(&self, side: Color, moves: &mut Vec<Move>) {
        let rights = self.castling_rights();
        let rank = match side {
            Color::White => 0,
            Color::Black => 7,
        };
        let king_square = Square::from_file_rank(4, rank);

        let (kingside_allowed, queenside_allowed) = match side {
            Color::White => (rights.white_kingside, rights.white_queenside),
            Color::Black => (rights.black_kingside, rights.black_queenside),
        };

        if kingside_allowed
            && self.squares_empty(&[
                Square::from_file_rank(5, rank),
                Square::from_file_rank(6, rank),
            ])
        {
            moves.push(Move::new(
                king_square,
                Square::from_file_rank(6, rank),
                MoveFlag::CastleKingside,
            ));
        }

        if queenside_allowed
            && self.squares_empty(&[
                Square::from_file_rank(1, rank),
                Square::from_file_rank(2, rank),
                Square::from_file_rank(3, rank),
            ])
        {
            moves.push(Move::new(
                king_square,
                Square::from_file_rank(2, rank),
                MoveFlag::CastleQueenside,
            ));
        }
    }

    fn squares_empty(&self, squares: &[Square]) -> bool {
        squares
            .iter()
            .all(|&square| self.piece_at(square).is_none())
    }

    fn occupied_by(&self, square: Square, color: Color) -> bool {
        matches!(self.piece_at(square), Some(Piece { color: c, .. }) if c == color)
    }
}

/// Pushes a pawn move to `to`, expanding it into all four promotion
/// moves if `to` is on the promotion rank, or a single quiet/capture
/// move otherwise. Whether it is a capture is implicit in whether `to`
/// was occupied by the opponent; that distinction is not encoded on
/// `Move` itself (see `moves.rs`).
fn push_pawn_move(from: Square, to: Square, promotion_rank: u8, moves: &mut Vec<Move>) {
    if to.rank() == promotion_rank {
        for &flag in &PROMOTION_KINDS {
            moves.push(Move::new(from, to, flag));
        }
    } else {
        moves.push(Move::new(from, to, MoveFlag::Quiet));
    }
}

/// Applies a file/rank offset to `square`, returning `None` if the
/// result falls off the board.
fn offset_square(square: Square, df: i8, dr: i8) -> Option<Square> {
    let file = square.file() as i8 + df;
    let rank = square.rank() as i8 + dr;
    if !(0..8).contains(&file) || !(0..8).contains(&rank) {
        return None;
    }
    Some(Square::from_file_rank(file as u8, rank as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_move(moves: &[Move], from: Square, to: Square, flag: MoveFlag) -> bool {
        moves
            .iter()
            .any(|mv| mv.from() == from && mv.to() == to && mv.flag() == flag)
    }

    fn sq(file: u8, rank: u8) -> Square {
        Square::from_file_rank(file, rank)
    }

    #[test]
    fn startpos_has_twenty_pseudo_legal_moves() {
        let position = Position::startpos();
        let moves = position.generate_pseudo_legal_moves();
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn startpos_pawn_has_single_and_double_push() {
        let position = Position::startpos();
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(&moves, sq(4, 1), sq(4, 2), MoveFlag::Quiet));
        assert!(contains_move(
            &moves,
            sq(4, 1),
            sq(4, 3),
            MoveFlag::DoublePawnPush
        ));
    }

    #[test]
    fn startpos_knight_moves_are_generated() {
        let position = Position::startpos();
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(&moves, sq(1, 0), sq(0, 2), MoveFlag::Quiet));
        assert!(contains_move(&moves, sq(1, 0), sq(2, 2), MoveFlag::Quiet));
    }

    #[test]
    fn pawn_blocked_has_no_push_moves() {
        let position = Position::from_fen("8/8/8/8/8/4p3/4P3/4K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(!moves.iter().any(|mv| mv.from() == sq(4, 1)));
    }

    #[test]
    fn pawn_double_push_blocked_by_piece_two_ahead() {
        let position = Position::from_fen("8/8/8/8/4n3/8/4P3/4K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(&moves, sq(4, 1), sq(4, 2), MoveFlag::Quiet));
        assert!(!contains_move(
            &moves,
            sq(4, 1),
            sq(4, 3),
            MoveFlag::DoublePawnPush
        ));
    }

    #[test]
    fn pawn_captures_diagonally() {
        let position = Position::from_fen("8/8/8/8/8/3p1p2/4P3/4K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(&moves, sq(4, 1), sq(3, 2), MoveFlag::Quiet));
        assert!(contains_move(&moves, sq(4, 1), sq(5, 2), MoveFlag::Quiet));
    }

    #[test]
    fn pawn_does_not_capture_own_piece() {
        let position = Position::from_fen("8/8/8/8/8/3P4/4P3/4K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(!contains_move(&moves, sq(4, 1), sq(3, 2), MoveFlag::Quiet));
    }

    #[test]
    fn pawn_promotes_on_last_rank() {
        let position = Position::from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        for flag in PROMOTION_KINDS {
            assert!(
                contains_move(&moves, sq(0, 6), sq(0, 7), flag),
                "missing promotion {flag:?}"
            );
        }
        assert_eq!(moves.iter().filter(|mv| mv.from() == sq(0, 6)).count(), 4);
    }

    #[test]
    fn pawn_promotes_on_capture() {
        let position = Position::from_fen("1n6/P7/8/8/8/8/8/4K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(
            &moves,
            sq(0, 6),
            sq(1, 7),
            MoveFlag::PromoteQueen
        ));
    }

    #[test]
    fn en_passant_capture_is_generated_when_available() {
        let position =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
                .expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(
            &moves,
            sq(4, 4),
            sq(3, 5),
            MoveFlag::EnPassant
        ));
    }

    #[test]
    fn en_passant_capture_absent_without_target_square() {
        let position = Position::startpos();
        let moves = position.generate_pseudo_legal_moves();
        assert!(!moves.iter().any(|mv| mv.flag() == MoveFlag::EnPassant));
    }

    #[test]
    fn bishop_slides_until_blocked() {
        let position = Position::from_fen("8/8/8/8/8/8/8/B3K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(&moves, sq(0, 0), sq(7, 7), MoveFlag::Quiet));
    }

    #[test]
    fn rook_stops_at_enemy_piece_and_can_capture_it() {
        let position = Position::from_fen("8/8/8/3p4/8/8/8/R3K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(&moves, sq(0, 0), sq(3, 0), MoveFlag::Quiet));
        assert!(!contains_move(&moves, sq(0, 0), sq(4, 0), MoveFlag::Quiet));
    }

    #[test]
    fn rook_does_not_pass_through_own_piece() {
        // White pawn on a2 blocks the a1 rook from advancing up the
        // a-file at all.
        let position = Position::from_fen("8/8/8/8/8/8/P7/R3K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(!moves
            .iter()
            .any(|mv| mv.from() == sq(0, 0) && mv.to().rank() >= 1));
        // It can still slide along the (empty) first rank.
        assert!(contains_move(&moves, sq(0, 0), sq(1, 0), MoveFlag::Quiet));
    }

    #[test]
    fn queen_moves_diagonally_and_orthogonally() {
        let position = Position::from_fen("8/8/8/8/8/8/8/3QK2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(&moves, sq(3, 0), sq(0, 0), MoveFlag::Quiet));
        assert!(contains_move(&moves, sq(3, 0), sq(3, 7), MoveFlag::Quiet));
        assert!(contains_move(&moves, sq(3, 0), sq(0, 3), MoveFlag::Quiet));
    }

    #[test]
    fn king_moves_one_square_each_direction() {
        let position = Position::from_fen("8/8/8/8/8/8/8/4K2k w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        let king_moves = moves.iter().filter(|mv| mv.from() == sq(4, 0)).count();
        assert_eq!(king_moves, 5); // corner-adjacent, so not all 8 fit on the board
    }

    #[test]
    fn king_can_castle_both_sides_when_rights_and_squares_allow() {
        let position = Position::from_fen("8/8/8/8/8/8/8/R3K2R w KQ - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(
            &moves,
            sq(4, 0),
            sq(6, 0),
            MoveFlag::CastleKingside
        ));
        assert!(contains_move(
            &moves,
            sq(4, 0),
            sq(2, 0),
            MoveFlag::CastleQueenside
        ));
    }

    #[test]
    fn castling_blocked_by_piece_between_king_and_rook() {
        let position = Position::from_fen("8/8/8/8/8/8/8/R2NK2R w KQ - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(!contains_move(
            &moves,
            sq(4, 0),
            sq(2, 0),
            MoveFlag::CastleQueenside
        ));
        assert!(contains_move(
            &moves,
            sq(4, 0),
            sq(6, 0),
            MoveFlag::CastleKingside
        ));
    }

    #[test]
    fn castling_not_generated_without_rights() {
        let position = Position::from_fen("8/8/8/8/8/8/8/R3K2R w - - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(!moves.iter().any(|mv| {
            mv.flag() == MoveFlag::CastleKingside || mv.flag() == MoveFlag::CastleQueenside
        }));
    }

    #[test]
    fn black_castling_moves_are_generated() {
        let position = Position::from_fen("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1").expect("valid FEN");
        let moves = position.generate_pseudo_legal_moves();
        assert!(contains_move(
            &moves,
            sq(4, 7),
            sq(6, 7),
            MoveFlag::CastleKingside
        ));
        assert!(contains_move(
            &moves,
            sq(4, 7),
            sq(2, 7),
            MoveFlag::CastleQueenside
        ));
    }

    #[test]
    fn generated_moves_apply_cleanly_via_make_move() {
        // Every pseudo-legal move from the starting position should be
        // applicable via make_move without panicking, and should be
        // reversible via unmake_move.
        let mut position = Position::startpos();
        let before = position.clone();
        for mv in position.clone().generate_pseudo_legal_moves() {
            let undo = position.make_move(mv);
            position.unmake_move(mv, undo);
            assert_eq!(position, before);
        }
    }
}
