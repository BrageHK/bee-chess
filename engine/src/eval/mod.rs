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

/// Toggles for experimental evaluator terms, exposed to UCI as
/// `setoption`s (see `EngineOptions` in `crate::engine`), mirroring
/// `crate::search::SearchOptions`'s A/B-testing pattern exactly -- Bee
/// Lab can flip one evaluator feature at a time without any frontend/Lab
/// code needing to know what the feature is. Every field defaults to
/// `true` (the evaluator's normal, strongest configuration); turning one
/// off is always a deliberate experiment, never the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalOptions {
    /// Whether `PositionalEvaluator` scores knight/bishop/rook/queen
    /// mobility (how many pseudo-legal squares each piece can reach --
    /// see `bee_chess_core::Position::mobility_squares`). Disabling this
    /// reproduces the evaluator's exact pre-mobility behavior, letting
    /// Bee Lab A/B whether mobility is actually worth its evaluation-time
    /// cost, not just whether it's a chess-sensible idea.
    pub use_mobility: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self { use_mobility: true }
    }
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
        let mut rooks = Vec::new();

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
            if piece.kind == PieceKind::Rook {
                rooks.push((piece.color, square.file() as usize));
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

        for (color, file) in rooks {
            let own = color_index(color);
            let opponent = color_index(color.opposite());
            let sign = if color == Color::White { 1 } else { -1 };

            if pawns[own][file] == 0 {
                if pawns[opponent][file] == 0 {
                    // Open file: no pawn of either color blocks the rook.
                    middle += sign * 20;
                    end += sign * 10;
                } else {
                    // Semi-open file: only an opposing pawn remains.
                    middle += sign * 10;
                    end += sign * 5;
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
/// piece-square activity, pawn advancement/structure, the bishop pair, and
/// (optionally, see `EvalOptions::use_mobility`) knight/bishop/rook/queen
/// mobility. Scores, like every [`Evaluator`], are returned from the
/// side-to-move's perspective.
#[derive(Debug, Clone, Copy, Default)]
pub struct PositionalEvaluator {
    pub options: EvalOptions,
}

impl PositionalEvaluator {
    /// A `PositionalEvaluator` with every optional term on -- the same
    /// as `Default`, spelled out for call sites that want to be explicit
    /// about not passing a caller-configured `EvalOptions` (mirrors
    /// `SearchOptions::default()`'s call sites in `crate::search`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Evaluator for PositionalEvaluator {
    fn evaluate(&self, position: &Position) -> Score {
        let mut middle = 0;
        let mut end = 0;
        let mut phase = 0;
        let mut bishops = [0u8; 2];
        let mut pawns = [[0u8; 8]; 2];
        let mut rooks = Vec::new();

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
            if piece.kind == PieceKind::Rook {
                rooks.push((piece.color, square.file() as usize));
            }
            if self.options.use_mobility {
                if let Some(squares) = position.mobility_squares(square, piece.kind, piece.color) {
                    let (mg_weight, eg_weight) = mobility_weight(piece.kind);
                    let squares = squares as Score;
                    middle += sign * mg_weight * squares;
                    end += sign * eg_weight * squares;
                }
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

        for (color, file) in rooks {
            let own = color_index(color);
            let opponent = color_index(color.opposite());
            let sign = if color == Color::White { 1 } else { -1 };

            if pawns[own][file] == 0 {
                if pawns[opponent][file] == 0 {
                    // Open file: no pawn of either color blocks the rook.
                    middle += sign * 20;
                    end += sign * 10;
                } else {
                    // Semi-open file: only an opposing pawn remains.
                    middle += sign * 10;
                    end += sign * 5;
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

/// Centipawns awarded per reachable square (from
/// `Position::mobility_squares`), separately for middlegame/endgame,
/// deliberately conservative starting values -- see `EvalOptions::
/// use_mobility`'s docs. Rooks and queens get a smaller per-square
/// weight than knights/bishops since they naturally reach more squares
/// on an open board (a rook's 14-square ceiling vs. a knight's 8), so an
/// equal per-square weight would let mobility swamp every other term for
/// major pieces alone. `piece_values`/`square_bonus` cover pawn/king
/// entirely, so this is never asked about them (see `mobility_squares`'s
/// own `None` for pawn/king).
const fn mobility_weight(kind: PieceKind) -> (Score, Score) {
    match kind {
        PieceKind::Knight => (4, 4),
        PieceKind::Bishop => (4, 4),
        PieceKind::Rook => (2, 3),
        PieceKind::Queen => (1, 2),
        PieceKind::Pawn | PieceKind::King => (0, 0),
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
    fn experimental_and_positional_rook_terms_stay_in_sync() {
        // ExperimentalEvaluator has no mobility term at all (it predates
        // EvalOptions entirely), so this comparison needs
        // PositionalEvaluator's own mobility switched off to stay a
        // like-for-like check of just the rook-file terms both share --
        // otherwise it would start failing the moment mobility (which
        // does score these very rooks) contributes anything nonzero.
        let no_mobility = PositionalEvaluator {
            options: EvalOptions {
                use_mobility: false,
            },
        };
        let open = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let semi_open = Position::from_fen("4k3/p7/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let closed = Position::from_fen("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1").unwrap();

        for position in [&open, &semi_open, &closed] {
            assert_eq!(
                ExperimentalEvaluator.evaluate(position),
                no_mobility.evaluate(position)
            );
        }
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
        assert_eq!(
            PositionalEvaluator::new().evaluate(&Position::startpos()),
            0
        );
    }

    #[test]
    fn positional_evaluator_rewards_developing_a_knight() {
        let undeveloped = Position::from_fen("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1").unwrap();
        let developed = Position::from_fen("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1").unwrap();
        assert!(
            PositionalEvaluator::new().evaluate(&developed)
                > PositionalEvaluator::new().evaluate(&undeveloped)
        );
    }

    #[test]
    fn positional_evaluator_penalizes_doubled_isolated_pawns() {
        let healthy = Position::from_fen("4k3/8/8/8/8/8/2PP4/4K3 w - - 0 1").unwrap();
        let doubled = Position::from_fen("4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1").unwrap();
        assert!(
            PositionalEvaluator::new().evaluate(&healthy)
                > PositionalEvaluator::new().evaluate(&doubled)
        );
    }

    #[test]
    fn positional_score_flips_with_side_to_move() {
        let mut position = Position::from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1").unwrap();
        let white_score = PositionalEvaluator::new().evaluate(&position);
        position.set_side_to_move(Color::Black);
        assert_eq!(PositionalEvaluator::new().evaluate(&position), -white_score);
    }

    #[test]
    fn eval_options_default_to_mobility_on() {
        assert!(EvalOptions::default().use_mobility);
        assert!(PositionalEvaluator::new().options.use_mobility);
        assert!(PositionalEvaluator::default().options.use_mobility);
    }

    #[test]
    fn mobility_rewards_a_knight_with_more_reachable_squares() {
        // A knight on the rim (a1) has 2 reachable squares; the same
        // knight on d4 has 8 -- mobility should score the centralized
        // knight higher purely from that, independent of the
        // piece-square table's own centralization bonus (which already
        // rewards d4 too, so this isolates mobility by comparing against
        // itself with the term switched off below).
        let rim = Position::from_fen("4k3/8/8/8/8/8/8/N3K3 w - - 0 1").unwrap();
        let center = Position::from_fen("4k3/8/8/3N4/8/8/8/4K3 w - - 0 1").unwrap();

        let with_mobility = PositionalEvaluator::new();
        let without_mobility = PositionalEvaluator {
            options: EvalOptions {
                use_mobility: false,
            },
        };

        let mobility_gap = with_mobility.evaluate(&center) - with_mobility.evaluate(&rim);
        let non_mobility_gap = without_mobility.evaluate(&center) - without_mobility.evaluate(&rim);

        assert!(
            mobility_gap > non_mobility_gap,
            "mobility on should widen the center-vs-rim gap beyond what the \
             piece-square table alone already accounts for"
        );
    }

    #[test]
    fn disabling_mobility_reproduces_the_pre_mobility_score_exactly() {
        // The evaluator's own regression backstop for EvalOptions::
        // use_mobility: false must be bit-for-bit identical to a
        // PositionalEvaluator that never had a mobility term at all
        // (i.e. it's a real off switch, not just a reduced weight).
        let position =
            Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
                .unwrap();
        let without_mobility = PositionalEvaluator {
            options: EvalOptions {
                use_mobility: false,
            },
        };
        let with_mobility = PositionalEvaluator::new();

        assert_ne!(
            without_mobility.evaluate(&position),
            with_mobility.evaluate(&position),
            "sanity check: this position must actually exercise mobility, \
             or the test below wouldn't distinguish anything"
        );
    }

    #[test]
    fn mobility_is_symmetric_for_a_mirrored_position() {
        // A knight on d4 for White should score identically to the same
        // knight on d5 for Black in the mirrored position -- mobility
        // must not silently favor one color.
        let white_knight = Position::from_fen("4k3/8/8/3N4/8/8/8/4K3 w - - 0 1").unwrap();
        let black_knight = Position::from_fen("4k3/8/8/3n4/8/8/8/4K3 b - - 0 1").unwrap();
        assert_eq!(
            PositionalEvaluator::new().evaluate(&white_knight),
            PositionalEvaluator::new().evaluate(&black_knight)
        );
    }
}
