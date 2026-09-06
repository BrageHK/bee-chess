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

use crate::chess::{Color, PieceKind, Position, Square};
use crate::search::Score;

/// Scores a position. Implementations must not perform network I/O or
/// other unbounded-latency work on this hot path (see CONTRIBUTING.md).
pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> Score;
}

/// A deliberately simple evaluator used for evaluation experiments.
pub struct ExperimentalEvaluator;

impl Evaluator for ExperimentalEvaluator {
    fn evaluate(&self, position: &Position) -> Score {
        let mut middle = 0;
        let mut end = 0;
        let mut phase = 0;
        let mut bishops = [0u8; 2];
        let mut pawns = [[0u8; 8]; 2];

        for index in 0..Square::COUNT as u8 {
            let square = Square::new(index);
            let Some(piece) = position.piece_at(square) else {
                continue;
            };
            let color = color_index(piece.color);
            let sign = if piece.color == Color::White { 1 } else { -1 };
            let relative_rank = if piece.color == Color::White {
                square.rank()
            } else {
                7 - square.rank()
            };

            let (mg_material, eg_material, phase_value) = piece_values(piece.kind);
            let (mg_square, eg_square) = square_bonus(piece.kind, square.file(), relative_rank);
            middle += sign * (mg_material + mg_square);
            end += sign * (eg_material + eg_square);
            phase += phase_value;

            if piece.kind == PieceKind::Bishop {
                bishops[color] += 1;
            }
            if piece.kind == PieceKind::Pawn {
                pawns[color][square.file() as usize] += 1;
            }
        }

        for color in [Color::White, Color::Black] {
            let i = color_index(color);
            let sign = if color == Color::White { 1 } else { -1 };
            if bishops[i] >= 2 {
                middle += sign * 30;
                end += sign * 45;
            }
            for file in 0..8 {
                let count = pawns[i][file];
                if count > 1 {
                    let extras = Score::from(count - 1);
                    middle -= sign * 12 * extras;
                    end -= sign * 18 * extras;
                }
                if count > 0
                    && (file == 0 || pawns[i][file - 1] == 0)
                    && (file == 7 || pawns[i][file + 1] == 0)
                {
                    middle -= sign * 10 * Score::from(count);
                    end -= sign * 8 * Score::from(count);
                }
            }
        }

        // Both armies together contribute 24 phase units initially: four
        // queens' worth, four rooks, and eight minor pieces at their weights.
        let phase = phase.min(24);
        let white_score = (middle * phase + end * (24 - phase)) / 24;
        let own_score = if position.side_to_move() == Color::White {
            white_score
        } else {
            -white_score
        };

        let own_mobility = position.generate_legal_moves().len() as Score;
        let mut opponent_position = position.clone();
        opponent_position.set_side_to_move(position.side_to_move().opposite());
        // An en-passant target is only valid for the actual side to move.
        opponent_position.set_en_passant_square(None);
        let opponent_mobility = opponent_position.generate_legal_moves().len() as Score;

        return own_score + 2 * (own_mobility - opponent_mobility);
    }
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

/// A tapered classical evaluator. Material and square activity are scored
/// separately for the middlegame and endgame, then blended according to the
/// non-pawn material left on the board. This keeps kings sheltered early but
/// makes them active once the major pieces have gone.
///
/// The deliberately small set of terms is cheap enough to run at every leaf:
/// piece-square activity, pawn advancement/structure, and the bishop pair.
/// Scores, like every [`Evaluator`], are returned from the side-to-move's
/// perspective.
pub struct PositionalEvaluator;

impl Evaluator for PositionalEvaluator {
    fn evaluate(&self, position: &Position) -> Score {
        let mut middle = 0;
        let mut end = 0;
        let mut phase = 0;
        let mut bishops = [0u8; 2];
        let mut pawns = [[0u8; 8]; 2];

        for index in 0..Square::COUNT as u8 {
            let square = Square::new(index);
            let Some(piece) = position.piece_at(square) else {
                continue;
            };
            let color = color_index(piece.color);
            let sign = if piece.color == Color::White { 1 } else { -1 };
            let relative_rank = if piece.color == Color::White {
                square.rank()
            } else {
                7 - square.rank()
            };

            let (mg_material, eg_material, phase_value) = piece_values(piece.kind);
            let (mg_square, eg_square) = square_bonus(piece.kind, square.file(), relative_rank);
            middle += sign * (mg_material + mg_square);
            end += sign * (eg_material + eg_square);
            phase += phase_value;

            if piece.kind == PieceKind::Bishop {
                bishops[color] += 1;
            }
            if piece.kind == PieceKind::Pawn {
                pawns[color][square.file() as usize] += 1;
            }
        }

        for color in [Color::White, Color::Black] {
            let i = color_index(color);
            let sign = if color == Color::White { 1 } else { -1 };
            if bishops[i] >= 2 {
                middle += sign * 30;
                end += sign * 45;
            }
            for file in 0..8 {
                let count = pawns[i][file];
                if count > 1 {
                    let extras = Score::from(count - 1);
                    middle -= sign * 12 * extras;
                    end -= sign * 18 * extras;
                }
                if count > 0
                    && (file == 0 || pawns[i][file - 1] == 0)
                    && (file == 7 || pawns[i][file + 1] == 0)
                {
                    middle -= sign * 10 * Score::from(count);
                    end -= sign * 8 * Score::from(count);
                }
            }
        }

        // Both armies together contribute 24 phase units initially: four
        // queens' worth, four rooks, and eight minor pieces at their weights.
        let phase = phase.min(24);
        let white_score = (middle * phase + end * (24 - phase)) / 24;
        if position.side_to_move() == Color::White {
            white_score
        } else {
            -white_score
        }
    }
}

const fn color_index(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

const fn piece_values(kind: PieceKind) -> (Score, Score, Score) {
    match kind {
        PieceKind::Pawn => (100, 120, 0),
        PieceKind::Knight => (320, 300, 1),
        PieceKind::Bishop => (330, 325, 1),
        PieceKind::Rook => (500, 525, 2),
        PieceKind::Queen => (900, 900, 4),
        PieceKind::King => (0, 0, 0),
    }
}

/// Compact, symmetric piece-square functions. `rank` is always measured from
/// the piece owner's home rank, making color symmetry explicit.
fn square_bonus(kind: PieceKind, file: u8, rank: u8) -> (Score, Score) {
    let file_distance = (file as Score - 3).abs().min((file as Score - 4).abs());
    let rank_distance = (rank as Score - 3).abs().min((rank as Score - 4).abs());
    let centre_distance = file_distance + rank_distance;
    match kind {
        PieceKind::Pawn => {
            let advance = Score::from(rank);
            (
                advance * 6 - file_distance * 3,
                advance * 12 - file_distance * 2,
            )
        }
        PieceKind::Knight => (30 - centre_distance * 12, 20 - centre_distance * 8),
        PieceKind::Bishop => (18 - centre_distance * 5, 22 - centre_distance * 4),
        PieceKind::Rook => (Score::from(rank == 6) * 20, Score::from(rank == 6) * 25),
        PieceKind::Queen => (8 - centre_distance * 3, 12 - centre_distance * 2),
        PieceKind::King => {
            // In the middlegame prefer the back rank and castled files; in
            // the ending reverse course and walk toward the centre.
            let castled_file_bonus = if file == 2 || file == 6 { 28 } else { 0 };
            (
                -Score::from(rank) * 12 + castled_file_bonus,
                35 - centre_distance * 12,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Color, Piece};

    #[test]
    fn experimental_start_position_is_balanced() {
        assert_eq!(ExperimentalEvaluator.evaluate(&Position::startpos()), 0);
    }

    #[test]
    fn experimental_evaluator_penalizes_isolated_pawns() {
        let connected = Position::from_fen("4k3/8/8/8/8/8/2PP4/4K3 w - - 0 1").unwrap();
        let isolated = Position::from_fen("4k3/8/8/8/8/8/2P2P2/4K3 w - - 0 1").unwrap();
        assert!(
            ExperimentalEvaluator.evaluate(&connected) > ExperimentalEvaluator.evaluate(&isolated)
        );
    }

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

    #[test]
    fn positional_start_position_is_symmetric() {
        assert_eq!(PositionalEvaluator.evaluate(&Position::startpos()), 0);
    }

    #[test]
    fn positional_evaluator_rewards_developing_a_knight() {
        let undeveloped = Position::from_fen("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1").unwrap();
        let developed = Position::from_fen("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1").unwrap();
        assert!(
            PositionalEvaluator.evaluate(&developed) > PositionalEvaluator.evaluate(&undeveloped)
        );
    }

    #[test]
    fn positional_evaluator_penalizes_doubled_isolated_pawns() {
        let healthy = Position::from_fen("4k3/8/8/8/8/8/2PP4/4K3 w - - 0 1").unwrap();
        let doubled = Position::from_fen("4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1").unwrap();
        assert!(PositionalEvaluator.evaluate(&healthy) > PositionalEvaluator.evaluate(&doubled));
    }

    #[test]
    fn positional_score_flips_with_side_to_move() {
        let mut position = Position::from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1").unwrap();
        let white_score = PositionalEvaluator.evaluate(&position);
        position.set_side_to_move(Color::Black);
        assert_eq!(PositionalEvaluator.evaluate(&position), -white_score);
    }
}
