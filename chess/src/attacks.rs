//! Attack detection and check-based legality.
//!
//! `is_square_attacked` answers "is this square attacked by any piece of
//! `by_color`", by looking outward from the target square along each
//! attack pattern rather than generating every move for every piece of
//! `by_color` and checking destinations — cheaper, and it doesn't care
//! whether `by_color` is the side to move (attack detection needs to
//! work for either side, and during legality checking it is always
//! asked about the side *not* to move).
//!
//! `generate_legal_moves` builds on this the simple way the milestone
//! plan asks for: generate pseudo-legal moves, make each one, check
//! whether the mover's own king is now attacked, and keep only the
//! moves where it isn't. This makes every legal move generation call
//! also do a full make/unmake per candidate move, which is slower than
//! the engine will eventually want — the milestone plan calls that
//! acceptable for now: prove correctness first, optimize later (e.g.
//! with a legality check that avoids the make/unmake round trip, or
//! incremental attack maps).

use super::movegen::{
    offset_square, BISHOP_DIRECTIONS, KING_OFFSETS, KNIGHT_OFFSETS, ROOK_DIRECTIONS,
};
use super::moves::{Move, MoveFlag};
use super::piece::{Color, PieceKind};
use super::position::Position;
use super::square::Square;

impl Position {
    /// Whether `square` is attacked by any piece of `by_color`.
    ///
    /// This does not care whether `by_color` is the side to move; it is
    /// a pure board query, which is what makes it reusable both for "is
    /// my king in check" (asking about the opponent's color) and for
    /// "would my king pass through an attacked square while castling"
    /// (same thing, at a different square).
    #[must_use]
    pub fn is_square_attacked(&self, square: Square, by_color: Color) -> bool {
        self.attacked_by_pawn(square, by_color)
            || self.attacked_by_knight(square, by_color)
            || self.attacked_by_king(square, by_color)
            || self.attacked_by_sliding(square, by_color, &BISHOP_DIRECTIONS, PieceKind::Bishop)
            || self.attacked_by_sliding(square, by_color, &ROOK_DIRECTIONS, PieceKind::Rook)
    }

    /// Whether the side to move's king is currently attacked.
    #[must_use]
    pub fn in_check(&self) -> bool {
        let side = self.side_to_move();
        match self.find_king(side) {
            Some(king_square) => self.is_square_attacked(king_square, side.opposite()),
            // A position with no king for the side to move is not a
            // legal chess position, but attack detection shouldn't
            // panic over it; treat it as "not in check" rather than
            // crashing search/perft on a malformed position.
            None => false,
        }
    }

    /// Generates all legal moves for the side to move: pseudo-legal
    /// moves, filtered down to those that don't leave the mover's own
    /// king attacked afterward.
    ///
    /// Castling moves get one extra check beyond "is the king attacked
    /// after the move": the square the king passes through (f1/f8 for
    /// kingside, d1/d8 for queenside) must also be unattacked, since
    /// make_move teleports the king directly to its landing square and
    /// would otherwise miss an attack on the transit square. Castling
    /// while currently in check is rejected too, since the king's
    /// starting square is checked as part of that same transit-square
    /// scan (kingside and queenside both transit through the square
    /// adjacent to the king).
    #[must_use]
    pub fn generate_legal_moves(&self) -> Vec<Move> {
        let mut position = self.clone();
        let side = self.side_to_move();
        let opponent = side.opposite();

        self.generate_pseudo_legal_moves()
            .into_iter()
            .filter(|&mv| {
                if let Some(transit_square) = castling_transit_square(mv) {
                    if position.is_square_attacked(mv.from(), opponent)
                        || position.is_square_attacked(transit_square, opponent)
                    {
                        return false;
                    }
                }

                let undo = position.make_move(mv);
                let king_square = position.find_king(side);
                let left_in_check = match king_square {
                    Some(square) => position.is_square_attacked(square, opponent),
                    None => false,
                };
                position.unmake_move(mv, undo);
                !left_in_check
            })
            .collect()
    }

    fn find_king(&self, color: Color) -> Option<Square> {
        (0..Square::COUNT as u8).map(Square::new).find(|&square| {
            matches!(
                self.piece_at(square),
                Some(piece) if piece.kind == PieceKind::King && piece.color == color
            )
        })
    }

    fn attacked_by_pawn(&self, square: Square, by_color: Color) -> bool {
        // A pawn of `by_color` attacks `square` if it sits one rank
        // behind (from the pawn's perspective) and one file to either
        // side -- i.e. looking from `square` backward along the pawn's
        // direction of travel.
        let behind: i8 = match by_color {
            Color::White => -1,
            Color::Black => 1,
        };
        [-1i8, 1i8].into_iter().any(|df| {
            offset_square(square, df, behind).is_some_and(|attacker_square| {
                matches!(
                    self.piece_at(attacker_square),
                    Some(piece) if piece.kind == PieceKind::Pawn && piece.color == by_color
                )
            })
        })
    }

    fn attacked_by_knight(&self, square: Square, by_color: Color) -> bool {
        KNIGHT_OFFSETS.iter().any(|&(df, dr)| {
            offset_square(square, df, dr).is_some_and(|attacker_square| {
                matches!(
                    self.piece_at(attacker_square),
                    Some(piece) if piece.kind == PieceKind::Knight && piece.color == by_color
                )
            })
        })
    }

    fn attacked_by_king(&self, square: Square, by_color: Color) -> bool {
        KING_OFFSETS.iter().any(|&(df, dr)| {
            offset_square(square, df, dr).is_some_and(|attacker_square| {
                matches!(
                    self.piece_at(attacker_square),
                    Some(piece) if piece.kind == PieceKind::King && piece.color == by_color
                )
            })
        })
    }

    /// Whether `square` is attacked by a sliding piece of `by_color`
    /// moving along `directions` -- either a piece of exactly `kind`,
    /// or a queen (which attacks along both bishop and rook lines).
    fn attacked_by_sliding(
        &self,
        square: Square,
        by_color: Color,
        directions: &[(i8, i8)],
        kind: PieceKind,
    ) -> bool {
        directions.iter().any(|&(df, dr)| {
            let mut current = square;
            while let Some(next) = offset_square(current, df, dr) {
                match self.piece_at(next) {
                    None => current = next,
                    Some(piece)
                        if piece.color == by_color
                            && (piece.kind == kind || piece.kind == PieceKind::Queen) =>
                    {
                        return true;
                    }
                    Some(_) => return false,
                }
            }
            false
        })
    }
}

/// If `mv` is a castling move, returns the square the king passes
/// through on its way to the destination (f1/f8 for kingside, d1/d8 for
/// queenside) -- the square that `is_square_attacked` must also check,
/// since `make_move` teleports the king straight to its landing square
/// and never visits this square itself.
fn castling_transit_square(mv: Move) -> Option<Square> {
    let rank = mv.from().rank();
    match mv.flag() {
        MoveFlag::CastleKingside => Some(Square::from_file_rank(5, rank)),
        MoveFlag::CastleQueenside => Some(Square::from_file_rank(3, rank)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(file: u8, rank: u8) -> Square {
        Square::from_file_rank(file, rank)
    }

    #[test]
    fn startpos_e4_is_attacked_by_neither_color() {
        let position = Position::startpos();
        assert!(!position.is_square_attacked(sq(4, 3), Color::White));
        assert!(!position.is_square_attacked(sq(4, 3), Color::Black));
    }

    #[test]
    fn startpos_no_one_is_in_check() {
        assert!(!Position::startpos().in_check());
    }

    #[test]
    fn pawn_attacks_diagonally_forward() {
        let position = Position::from_fen("8/8/8/8/8/1p6/8/4K2k w - - 0 1").expect("valid FEN");
        assert!(position.is_square_attacked(sq(0, 1), Color::Black));
        assert!(position.is_square_attacked(sq(2, 1), Color::Black));
        assert!(!position.is_square_attacked(sq(1, 1), Color::Black));
    }

    #[test]
    fn knight_attacks_l_shape() {
        let position = Position::from_fen("8/8/8/3n4/8/8/8/4K2k w - - 0 1").expect("valid FEN");
        assert!(position.is_square_attacked(sq(1, 3), Color::Black)); // b4
        assert!(position.is_square_attacked(sq(5, 3), Color::Black)); // f4
        assert!(!position.is_square_attacked(sq(3, 4), Color::Black)); // d5 itself
        assert!(!position.is_square_attacked(sq(1, 4), Color::Black)); // b5: not a knight move
    }

    #[test]
    fn king_attacks_adjacent_squares() {
        // Black king on e5 (file 4, rank 4).
        let position = Position::from_fen("8/8/8/4k3/8/8/8/7K w - - 0 1").expect("valid FEN");
        assert!(position.is_square_attacked(sq(4, 5), Color::Black)); // e6, adjacent
        assert!(position.is_square_attacked(sq(3, 3), Color::Black)); // d4, adjacent
        assert!(!position.is_square_attacked(sq(4, 4), Color::Black)); // e5 itself
        assert!(!position.is_square_attacked(sq(4, 2), Color::Black)); // e3, two ranks away
    }

    #[test]
    fn rook_attacks_along_rank_and_file() {
        let position = Position::from_fen("8/8/8/3r4/8/8/8/7K w - - 0 1").expect("valid FEN");
        assert!(position.is_square_attacked(sq(3, 0), Color::Black));
        assert!(position.is_square_attacked(sq(3, 7), Color::Black));
        assert!(position.is_square_attacked(sq(0, 4), Color::Black));
        assert!(!position.is_square_attacked(sq(4, 5), Color::Black)); // not on rank/file
    }

    #[test]
    fn rook_attack_blocked_by_intervening_piece() {
        let position = Position::from_fen("8/8/8/3r4/3P4/8/8/7K w - - 0 1").expect("valid FEN");
        assert!(!position.is_square_attacked(sq(3, 0), Color::Black));
    }

    #[test]
    fn bishop_attacks_diagonally() {
        // Black bishop on d5 (file 3, rank 4).
        let position = Position::from_fen("8/8/8/3b4/8/8/8/7K w - - 0 1").expect("valid FEN");
        assert!(position.is_square_attacked(sq(0, 1), Color::Black)); // a2
        assert!(position.is_square_attacked(sq(6, 7), Color::Black)); // g8
        assert!(!position.is_square_attacked(sq(3, 0), Color::Black)); // d1, not diagonal
    }

    #[test]
    fn queen_attacks_like_rook_and_bishop() {
        let position = Position::from_fen("8/8/8/3q4/8/8/8/7K w - - 0 1").expect("valid FEN");
        assert!(position.is_square_attacked(sq(3, 0), Color::Black)); // file
        assert!(position.is_square_attacked(sq(0, 4), Color::Black)); // rank
        assert!(position.is_square_attacked(sq(0, 1), Color::Black)); // diagonal
    }

    #[test]
    fn in_check_true_when_king_attacked() {
        let position = Position::from_fen("8/8/8/8/8/8/8/r3K2k w - - 0 1").expect("valid FEN");
        assert!(position.in_check());
    }

    #[test]
    fn in_check_false_when_king_safe() {
        let position = Position::from_fen("8/8/8/8/8/8/8/4K2k w - - 0 1").expect("valid FEN");
        assert!(!position.in_check());
    }

    #[test]
    fn in_check_only_considers_side_to_move() {
        // Black rook attacks a1, but it is White's king there and
        // White to move -- in_check() should report White's status.
        let position = Position::from_fen("8/8/8/8/8/8/8/r3K2k w - - 0 1").expect("valid FEN");
        assert!(position.in_check());

        // Flip side to move: now it's Black's king (h1) we care about,
        // which is not attacked.
        let position = Position::from_fen("8/8/8/8/8/8/8/r3K2k b - - 0 1").expect("valid FEN");
        assert!(!position.in_check());
    }

    #[test]
    fn startpos_has_twenty_legal_moves() {
        let moves = Position::startpos().generate_legal_moves();
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn pinned_piece_cannot_move_and_expose_king() {
        // White king on e1, white rook pinned on e2 by a black rook on
        // e8, along the e-file. The pinned rook cannot move off the
        // e-file without exposing the king to check.
        let position = Position::from_fen("4r3/8/8/8/8/8/4R3/4K3 w - - 0 1").expect("valid FEN");
        let moves = position.generate_legal_moves();
        assert!(!moves
            .iter()
            .any(|mv| mv.from() == sq(4, 1) && mv.to().file() != 4));
        // It can still move along the pin line.
        assert!(moves
            .iter()
            .any(|mv| mv.from() == sq(4, 1) && mv.to() == sq(4, 2)));
    }

    #[test]
    fn king_cannot_move_into_check() {
        // White king on e1; black rook on e8 controls the whole e-file,
        // so the king cannot step to e2, but can step sideways.
        let position = Position::from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1").expect("valid FEN");
        let moves = position.generate_legal_moves();
        assert!(!moves
            .iter()
            .any(|mv| mv.from() == sq(4, 0) && mv.to() == sq(4, 1)));
        assert!(moves
            .iter()
            .any(|mv| mv.from() == sq(4, 0) && mv.to() == sq(3, 0)));
    }

    #[test]
    fn only_moves_that_resolve_check_are_legal_when_in_check() {
        // White king on e1 in check from a black rook on e8. Moving an
        // unrelated piece (the a1 rook, sideways) does nothing about
        // the check and must be filtered out as illegal, even though
        // it's pseudo-legal.
        let position = Position::from_fen("4r3/8/8/8/8/8/8/R3K3 w - - 0 1").expect("valid FEN");
        let moves = position.generate_legal_moves();
        assert!(!moves
            .iter()
            .any(|mv| mv.from() == sq(0, 0) && mv.to() == sq(1, 0)));
        // The king stepping off the e-file to a safe square is legal.
        assert!(moves
            .iter()
            .any(|mv| mv.from() == sq(4, 0) && mv.to() == sq(3, 0)));
    }

    #[test]
    fn checkmate_position_has_no_legal_moves() {
        // Classic back-rank mate: white king on h1 boxed in by its own
        // pawns on f2/g2/h2, black rook on a1 delivering check along
        // the first rank. The king's only nominal escape (g1) is still
        // on the attacked rank, and nothing can block or capture the
        // rook.
        let position = Position::from_fen("6k1/8/8/8/8/8/5PPP/r6K w - - 0 1").expect("valid FEN");
        assert!(position.in_check());
        assert!(position.generate_legal_moves().is_empty());
    }

    #[test]
    fn castling_through_check_is_illegal() {
        // White king on e1, rook on h1, all squares empty between them,
        // castling rights present, but a black rook on f8 attacks f1 --
        // the square the king passes through -- so kingside castling
        // must be filtered out even though it's pseudo-legal.
        let position = Position::from_fen("4k3/5r2/8/8/8/8/8/4K2R w K - 0 1").expect("valid FEN");
        let moves = position.generate_legal_moves();
        assert!(!moves.iter().any(|mv| mv.flag() == MoveFlag::CastleKingside));
    }

    #[test]
    fn castling_while_in_check_is_illegal() {
        // White king on e1 is currently in check from a rook on e8;
        // castling is pseudo-legal (rights held, squares between king
        // and rook empty) but must not be legal while in check.
        let position = Position::from_fen("4r3/8/8/8/8/8/8/4K2R w K - 0 1").expect("valid FEN");
        let moves = position.generate_legal_moves();
        assert!(!moves.iter().any(|mv| mv.flag() == MoveFlag::CastleKingside));
    }

    #[test]
    fn castling_into_check_is_illegal() {
        // King would land on g1, which is attacked by a black rook on
        // g8. Castling rights and path are otherwise clear.
        let position = Position::from_fen("4k1r1/8/8/8/8/8/8/4K2R w K - 0 1").expect("valid FEN");
        let moves = position.generate_legal_moves();
        assert!(!moves.iter().any(|mv| mv.flag() == MoveFlag::CastleKingside));
    }

    #[test]
    fn legal_castling_remains_when_nothing_is_attacked() {
        let position = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").expect("valid FEN");
        let moves = position.generate_legal_moves();
        assert!(moves.iter().any(|mv| mv.flag() == MoveFlag::CastleKingside));
    }
}
