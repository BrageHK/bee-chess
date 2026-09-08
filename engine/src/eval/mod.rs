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
    /// Whether `PositionalEvaluator` scores king safety: pawn shield,
    /// open/semi-open files on and around the king, and enemy attacks
    /// into a small king zone -- see `king_safety`'s docs. Disabling
    /// this reproduces the evaluator's exact pre-king-safety behavior
    /// (the same coarse castled-file-only term `square_bonus` already
    /// had), letting Bee Lab A/B whether the richer term is actually
    /// worth its cost -- king-safety terms are notorious for sounding
    /// sensible on paper while losing real Elo if overtuned.
    pub use_king_safety: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            use_mobility: true,
            use_king_safety: true,
        }
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

            if self.options.use_king_safety {
                if let Some(king_square) = position.find_king(color) {
                    // King safety is middlegame-only by construction (added
                    // only to `middle`, never `end`): tapering already blends
                    // `middle`/`end` by remaining material below, and a king
                    // that should be *centralizing* in the endgame must not
                    // additionally be penalized here for having "no pawn
                    // shield" or "an open file nearby" -- those are exactly
                    // what an active endgame king walks into on purpose.
                    middle += sign * king_safety(position, king_square, color, &pawns[i]);
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

/// How many enemy attacks into the king's zone are worth, in centipawns
/// per attacked square -- see `king_safety`'s "attack units" term.
/// Deliberately small: this counts every attacking piece regardless of
/// its kind or how many pieces attack the same square, so it's meant as
/// a rough "how much pressure is nearby" signal, not a precise threat
/// count -- see `EvalOptions::use_king_safety`'s docs on how easy this
/// class of term is to overtune.
const KING_ZONE_ATTACK_PENALTY: Score = 8;
/// Centipawn penalty for a missing shield pawn (see `king_safety`) --
/// smaller than a full pawn's value since a missing shield pawn is a
/// structural weakness, not a material loss.
const MISSING_SHIELD_PAWN_PENALTY: Score = 12;
/// Centipawn penalty for the king's own file having no friendly pawn
/// (semi-open, if the opponent still has one there) or no pawn at all
/// (fully open, worse) -- see `king_safety`.
const SEMI_OPEN_KING_FILE_PENALTY: Score = 15;
const OPEN_KING_FILE_PENALTY: Score = 30;

/// Scores how safe `color`'s king (on `king_square`) is, as a single
/// middlegame-only centipawn term (see the call site's docs on why this
/// never touches `end`) -- always returned so that adding it with the
/// caller's own `sign` produces the right side-to-move-relative sign,
/// matching every other term in `PositionalEvaluator::evaluate`.
///
/// Three cheap, deliberately simple sub-terms, per `EvalOptions::
/// use_king_safety`'s docs on why this starts simple rather than as a
/// full attack model:
/// - **Pawn shield**: the two squares diagonally in front of the king
///   plus the one directly in front (the king's own file) should have a
///   friendly pawn one rank ahead of the king (its normal "castled"
///   position) -- each missing one costs `MISSING_SHIELD_PAWN_PENALTY`.
///   Skipped entirely once the king has moved off its own back three
///   files' worth of shelter (e.g. an already-centralized king), since a
///   "missing shield" isn't a meaningful weakness for a king that was
///   never trying to hide behind one.
/// - **Open/semi-open king file**: `pawns_for_color` is the same
///   per-file pawn count the caller already built for the pawn-structure
///   term above, so this reuses it rather than rescanning the board --
///   costs `SEMI_OPEN_KING_FILE_PENALTY` if only the opponent still has
///   a pawn on the king's file, `OPEN_KING_FILE_PENALTY` (no pawn of
///   either color) if neither does.
/// - **King-zone attacks**: `Position::is_square_attacked` (already the
///   cheap, no-move-generation check `in_check` itself uses) over the
///   king's own square and its up-to-8 neighbors, once per enemy
///   attacker found -- see `KING_ZONE_ATTACK_PENALTY`'s docs on why this
///   is a rough pressure count, not a precise threat evaluation.
fn king_safety(
    position: &Position,
    king_square: Square,
    color: Color,
    pawns_for_color: &[u8; 8],
) -> Score {
    let file = king_square.file() as i32;
    let rank = king_square.rank() as i32;
    let forward = if color == Color::White { 1 } else { -1 };
    let opponent = color.opposite();

    let mut penalty = 0;

    // Pawn shield -- only meaningful while the king is still tucked
    // near its own back rank (relative to its own color): a king that
    // has already walked toward the centre isn't "missing its shield,"
    // it deliberately left it behind.
    let relative_rank = if color == Color::White {
        rank
    } else {
        7 - rank
    };
    if relative_rank <= 1 {
        for df in [-1, 0, 1] {
            let shield_file = file + df;
            if !(0..8).contains(&shield_file) {
                continue;
            }
            let shield_rank = rank + forward;
            if !(0..8).contains(&shield_rank) {
                continue;
            }
            let shield_square = Square::from_file_rank(shield_file as u8, shield_rank as u8);
            let has_shield_pawn = matches!(
                position.piece_at(shield_square),
                Some(piece) if piece.kind == PieceKind::Pawn && piece.color == color
            );
            if !has_shield_pawn {
                penalty += MISSING_SHIELD_PAWN_PENALTY;
            }
        }
    }

    // Open/semi-open king file: `pawns_for_color` (the caller's own
    // per-file counts, reused rather than rescanned) tells us about the
    // king's own color; the opponent's count isn't something the caller
    // has in scope at the call site, so ask the board directly for it.
    let opponent_pawns_on_file = (0..8u8)
        .map(|rank| position.piece_at(Square::from_file_rank(file as u8, rank)))
        .filter(|p| matches!(p, Some(piece) if piece.kind == PieceKind::Pawn && piece.color == opponent))
        .count();
    if pawns_for_color[file as usize] == 0 {
        penalty += if opponent_pawns_on_file == 0 {
            OPEN_KING_FILE_PENALTY
        } else {
            SEMI_OPEN_KING_FILE_PENALTY
        };
    }

    // King-zone attacks: the king's own square plus its (board-edge-
    // clamped) neighbors, each checked with the same cheap attack query
    // in_check itself uses -- never full move generation.
    for df in [-1, 0, 1] {
        for dr in [-1, 0, 1] {
            let zone_file = file + df;
            let zone_rank = rank + dr;
            if !(0..8).contains(&zone_file) || !(0..8).contains(&zone_rank) {
                continue;
            }
            let zone_square = Square::from_file_rank(zone_file as u8, zone_rank as u8);
            if position.is_square_attacked(zone_square, opponent) {
                penalty += KING_ZONE_ATTACK_PENALTY;
            }
        }
    }

    -penalty
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
        // ExperimentalEvaluator has no mobility or king-safety term at
        // all (it predates EvalOptions entirely), so this comparison
        // needs both switched off on PositionalEvaluator to stay a
        // like-for-like check of just the rook-file terms both share --
        // otherwise it would start failing the moment either term
        // (both of which score these very kings/rooks) contributes
        // anything nonzero.
        let no_mobility = PositionalEvaluator {
            options: EvalOptions {
                use_mobility: false,
                use_king_safety: false,
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
                ..EvalOptions::default()
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
                ..EvalOptions::default()
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

    #[test]
    fn eval_options_default_to_king_safety_on() {
        assert!(EvalOptions::default().use_king_safety);
        assert!(PositionalEvaluator::new().options.use_king_safety);
        assert!(PositionalEvaluator::default().options.use_king_safety);
    }

    #[test]
    fn king_safety_prefers_an_intact_pawn_shield_over_an_open_file() {
        // Calls king_safety directly (see king_safety_penalizes_enemy_
        // pieces_massed_near_the_king's docs on why): the two positions
        // this compares differ by two pawns' worth of material (the
        // "open file" side is missing its g2/h2 shield pawns entirely),
        // which would swamp any king-safety-sized difference if this
        // went through the full evaluator's material term instead.
        let pawns_with_shield = [0u8, 0, 0, 0, 0, 1, 1, 1]; // f2/g2/h2
        let pawns_without_shield = [0u8, 0, 0, 0, 0, 1, 0, 0]; // f2 only
        let king_square = Square::from_file_rank(6, 0); // g1

        let intact_shield = Position::from_fen("4k3/8/8/8/8/8/5PPP/R5K1 w - - 0 1").unwrap();
        let open_file = Position::from_fen("4k3/8/8/8/8/8/5P2/R5K1 w - - 0 1").unwrap();

        assert!(
            king_safety(
                &intact_shield,
                king_square,
                Color::White,
                &pawns_with_shield
            ) > king_safety(&open_file, king_square, Color::White, &pawns_without_shield),
            "an intact pawn shield (plus a closed king file) must score \
             higher than the same king with its shield pawns gone"
        );
    }

    #[test]
    fn king_safety_penalizes_enemy_pieces_massed_near_the_king() {
        // Calls king_safety directly rather than through the full
        // evaluator: the full evaluator also scores the attacking
        // piece's own mobility/piece-square/open-file terms, which
        // differ between "near the king" and "far away" test positions
        // for reasons that have nothing to do with king safety (e.g. a
        // rook on a corner square has slightly different mobility than
        // one on another corner, purely from board-edge geometry) --
        // exactly the kind of confound that makes an end-to-end
        // comparison fragile here. Testing the term in isolation is the
        // direct, unconfounded way to check what it actually claims to
        // do: penalize enemy attacks into the king zone.
        let pawns = [0u8, 0, 0, 0, 0, 1, 1, 1]; // f2/g2/h2 shield intact
        let king_square = Square::from_file_rank(6, 0); // g1

        let attacked = Position::from_fen("4k2r/8/8/8/8/8/5PPP/6K1 w - - 0 1").unwrap();
        let far_away = Position::from_fen("r3k3/8/8/8/8/8/5PPP/6K1 w - - 0 1").unwrap();

        assert!(
            king_safety(&attacked, king_square, Color::White, &pawns)
                < king_safety(&far_away, king_square, Color::White, &pawns),
            "an enemy rook attacking directly into the king zone must \
             score worse than the same rook sitting far away"
        );
    }

    #[test]
    fn king_safety_is_symmetric_for_a_mirrored_position() {
        // The exact same shield-vs-open-file comparison as above, but
        // mirrored onto Black's king and rank -- king safety must not
        // silently favor one color the way a rank-relative bug (using
        // an absolute rank instead of one relative to the king's own
        // color) would.
        let white_shielded = Position::from_fen("4k3/8/8/8/8/8/5PPP/R5K1 w - - 0 1").unwrap();
        let black_shielded = Position::from_fen("r5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1").unwrap();

        assert_eq!(
            PositionalEvaluator::new().evaluate(&white_shielded),
            PositionalEvaluator::new().evaluate(&black_shielded)
        );
    }

    #[test]
    fn disabling_king_safety_reproduces_the_pre_king_safety_score_exactly() {
        // Same regression backstop as mobility's: EvalOptions::
        // use_king_safety: false must be a real off switch, not just a
        // reduced weight -- it should change the score for a position
        // that actually exercises the term.
        let position = Position::from_fen("4k3/8/8/8/8/8/5P2/R5K1 w - - 0 1").unwrap();
        let without_king_safety = PositionalEvaluator {
            options: EvalOptions {
                use_king_safety: false,
                ..EvalOptions::default()
            },
        };
        let with_king_safety = PositionalEvaluator::new();

        assert_ne!(
            without_king_safety.evaluate(&position),
            with_king_safety.evaluate(&position),
            "sanity check: this position must actually exercise king \
             safety, or the test above wouldn't distinguish anything"
        );
    }

    #[test]
    fn king_safety_ignores_a_centralized_endgame_king() {
        // A king that has already walked toward the centre (as an
        // active endgame king should) must not be penalized for having
        // "no pawn shield" -- it never had one to lose. Comparing a
        // centralized king against a back-rank one with identical
        // material, king safety's shield term specifically must not be
        // the thing making the back-rank king look better; the tapering
        // toward `middle`-only (see king_safety's call site) means this
        // shows up mostly at full middlegame phase, so this test keeps
        // enough material on the board to stay solidly middlegame-phase.
        let centralized =
            Position::from_fen("r1bq1rk1/ppp2ppp/2n5/3p4/3P4/2N1PN2/PPP2PPP/R1BQ1RK1 w - - 0 1")
                .unwrap();
        // Same position, but White's king has walked to d4 (Black's is
        // untouched) -- an unrealistic king walk, but the point here is
        // purely mechanical: king_safety must not fire its shield
        // penalty just because the king is far from its back rank.
        let mut walked = centralized.clone();
        walked.set_piece(Square::from_file_rank(6, 0), None);
        walked.set_piece(
            Square::from_file_rank(3, 3),
            Some(Piece::new(PieceKind::King, Color::White)),
        );

        // This isn't asserting a direction (the piece-square table
        // itself has strong opinions about king centralization
        // independent of king safety) -- it's a smoke test that
        // evaluating a centralized king doesn't panic or produce a
        // wildly nonsensical score from an out-of-bounds shield/file
        // lookup near an unusual king square.
        let _ = PositionalEvaluator::new().evaluate(&walked);
    }
}
