//! Evaluator contract and the first concrete evaluator.
//!
//! An `Evaluator` scores a position from the side-to-move's perspective
//! (i.e. positive always means "good for whoever is about to move" --
//! this is what lets negamax negate scores uniformly instead of
//! branching on color). Per ADR 0001, the v1 evaluator is eventually an
//! incrementally updatable neural evaluator; `MaterialEvaluator` is the
//! first slice, used to get alpha-beta search itself correct before any
//! evaluation sophistication. Concrete evaluators (NNUE, an ONNX
//! reference backend) are implemented in follow-up PRs behind this same
//! trait, without the search architecture needing to change.

use crate::chess::{PieceKind, Position, Square};
use crate::search::Score;

/// Scores a position. Implementations must not perform network I/O or
/// other unbounded-latency work on this hot path (see CONTRIBUTING.md).
pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> Score;
}

/// Standard piece values in centipawns, from the classical scale
/// (pawn=100, knight=320, bishop=330, rook=500, queen=900). No
/// positional terms, no king safety, no pawn structure -- purely
/// material, since the goal of this PR is a correct alpha-beta search,
/// not a strong evaluator.
const PAWN_VALUE: Score = 100;
const KNIGHT_VALUE: Score = 320;
const BISHOP_VALUE: Score = 330;
const ROOK_VALUE: Score = 500;
const QUEEN_VALUE: Score = 900;
/// The king is never captured (search stops at checkmate before that
/// could happen), so it contributes nothing to material score.
const KING_VALUE: Score = 0;

fn piece_value(kind: PieceKind) -> Score {
    match kind {
        PieceKind::Pawn => PAWN_VALUE,
        PieceKind::Knight => KNIGHT_VALUE,
        PieceKind::Bishop => BISHOP_VALUE,
        PieceKind::Rook => ROOK_VALUE,
        PieceKind::Queen => QUEEN_VALUE,
        PieceKind::King => KING_VALUE,
    }
}

/// Evaluates a position purely by material balance: the sum of the
/// side-to-move's own piece values minus their opponent's, so the
/// result is already in the side-to-move-relative form `Evaluator`
/// requires.
pub struct MaterialEvaluator;

impl Evaluator for MaterialEvaluator {
    fn evaluate(&self, position: &Position) -> Score {
        let side = position.side_to_move();
        let mut score: Score = 0;

        for index in 0..Square::COUNT as u8 {
            let Some(piece) = position.piece_at(Square::new(index)) else {
                continue;
            };
            let value = piece_value(piece.kind);
            score += if piece.color == side { value } else { -value };
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Color, Piece};

    #[test]
    fn startpos_is_exactly_balanced() {
        let evaluator = MaterialEvaluator;
        assert_eq!(evaluator.evaluate(&Position::startpos()), 0);
    }

    #[test]
    fn empty_board_is_balanced() {
        let evaluator = MaterialEvaluator;
        assert_eq!(evaluator.evaluate(&Position::empty()), 0);
    }

    #[test]
    fn favors_the_side_to_move_when_material_up() {
        // White has an extra queen; White to move should see a large
        // positive score.
        let mut position = Position::empty();
        position.set_piece(
            Square::from_file_rank(0, 0),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        position.set_piece(
            Square::from_file_rank(7, 7),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        position.set_piece(
            Square::from_file_rank(3, 3),
            Some(Piece::new(PieceKind::Queen, Color::White)),
        );

        let evaluator = MaterialEvaluator;
        assert_eq!(evaluator.evaluate(&position), QUEEN_VALUE);
    }

    #[test]
    fn score_flips_sign_with_side_to_move() {
        // Same material imbalance (White up a queen), but Black to
        // move: the score must be negative, since it's always
        // relative to whoever is about to move.
        let mut position = Position::empty();
        position.set_piece(
            Square::from_file_rank(0, 0),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        position.set_piece(
            Square::from_file_rank(7, 7),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        position.set_piece(
            Square::from_file_rank(3, 3),
            Some(Piece::new(PieceKind::Queen, Color::White)),
        );
        position.set_side_to_move(Color::Black);

        let evaluator = MaterialEvaluator;
        assert_eq!(evaluator.evaluate(&position), -QUEEN_VALUE);
    }

    #[test]
    fn sums_multiple_pieces_correctly() {
        let mut position = Position::empty();
        position.set_piece(
            Square::from_file_rank(0, 0),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        position.set_piece(
            Square::from_file_rank(7, 7),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        // White: rook + bishop. Black: knight.
        position.set_piece(
            Square::from_file_rank(1, 1),
            Some(Piece::new(PieceKind::Rook, Color::White)),
        );
        position.set_piece(
            Square::from_file_rank(2, 2),
            Some(Piece::new(PieceKind::Bishop, Color::White)),
        );
        position.set_piece(
            Square::from_file_rank(5, 5),
            Some(Piece::new(PieceKind::Knight, Color::Black)),
        );

        let evaluator = MaterialEvaluator;
        let expected = ROOK_VALUE + BISHOP_VALUE - KNIGHT_VALUE;
        assert_eq!(evaluator.evaluate(&position), expected);
    }
}
