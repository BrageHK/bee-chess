//! Fixed-depth negamax alpha-beta search: the first slice of #6.
//!
//! Deliberately excludes (see later #6 PRs): quiescence search (so it
//! suffers the horizon effect at the search boundary), iterative
//! deepening (search goes straight to the requested depth), a
//! transposition table, move ordering, aspiration windows, and PVS.
//! None of those affect correctness at a fixed depth; they only affect
//! speed and (for quiescence) tactical horizon quality. Get the tree
//! itself correct first.

use crate::chess::Position;
use crate::eval::Evaluator;

use super::{Score, SearchResult, SCORE_INF, SCORE_MATE};

/// Searches `position` to exactly `depth` plies using negamax
/// alpha-beta, scoring leaves with `evaluator`. Returns the best move
/// found at the root (or `None` if the root position is checkmate or
/// stalemate), the score of that line from the root side to move's
/// perspective, and the total number of nodes visited.
///
/// `position` is restored to its exact starting state before this
/// returns: every recursive `make_move` is paired with an `unmake_move`
/// on every path, including the ones alpha-beta cuts off early.
pub fn search(position: &mut Position, depth: u32, evaluator: &impl Evaluator) -> SearchResult {
    let mut nodes = 0u64;
    let moves = position.generate_legal_moves();

    if moves.is_empty() {
        // Root is checkmate or stalemate: nothing to play, but still a
        // well-defined score.
        let score = terminal_score(position, 0);
        return SearchResult {
            best_move: None,
            score,
            nodes: 1,
        };
    }

    let mut best_move = moves[0];
    let mut best_score = -SCORE_INF;
    let mut alpha = -SCORE_INF;
    let beta = SCORE_INF;

    for mv in moves {
        let undo = position.make_move(mv);
        let score = -negamax(position, depth - 1, -beta, -alpha, 1, evaluator, &mut nodes);
        position.unmake_move(mv, undo);

        if score > best_score {
            best_score = score;
            best_move = mv;
        }
        alpha = alpha.max(score);
        // No beta cutoff at the root: we need to have actually
        // compared every move to know which one is best, not just
        // that some move is "good enough."
    }

    SearchResult {
        best_move: Some(best_move),
        score: best_score,
        nodes: nodes + 1, // +1 for the root position itself
    }
}

/// The recursive negamax search. `ply` is the distance from the root,
/// used only to ply-adjust mate scores (see `SCORE_MATE`'s docs) so a
/// shorter forced mate is always preferred over a longer one and a
/// losing side delays mate as long as possible.
fn negamax(
    position: &mut Position,
    depth: u32,
    mut alpha: Score,
    beta: Score,
    ply: u32,
    evaluator: &impl Evaluator,
    nodes: &mut u64,
) -> Score {
    *nodes += 1;

    let moves = position.generate_legal_moves();
    if moves.is_empty() {
        return terminal_score(position, ply);
    }

    if depth == 0 {
        return evaluator.evaluate(position);
    }

    let mut best = -SCORE_INF;

    for mv in moves {
        let undo = position.make_move(mv);
        let score = -negamax(
            position,
            depth - 1,
            -beta,
            -alpha,
            ply + 1,
            evaluator,
            nodes,
        );
        position.unmake_move(mv, undo);

        best = best.max(score);
        alpha = alpha.max(score);

        if alpha >= beta {
            break; // beta cutoff: the opponent won't allow this line
        }
    }

    best
}

/// The score for a position with no legal moves: checkmate (ply-
/// adjusted, from the perspective of the side to move, who is being
/// mated) or stalemate (an exact draw).
fn terminal_score(position: &Position, ply: u32) -> Score {
    if position.in_check() {
        -SCORE_MATE + ply as Score
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::Position;
    use crate::eval::MaterialEvaluator;
    use crate::search::mate_in_plies;

    #[test]
    fn finds_mate_in_one() {
        // Black king g8 boxed in by its own pawns on f7/g7/h7; Qd1-d8
        // delivers back-rank mate.
        let mut position =
            Position::from_fen("6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1").expect("valid FEN");

        let result = search(&mut position, 2, &MaterialEvaluator);

        let best_move = result.best_move.expect("should find a move");
        assert_eq!(best_move.from(), "d1".parse().unwrap());
        assert_eq!(best_move.to(), "d8".parse().unwrap());
        assert_eq!(mate_in_plies(result.score), Some(1));
    }

    #[test]
    fn state_is_fully_restored_after_search() {
        let mut position = Position::startpos();
        let before = position.clone();

        search(&mut position, 4, &MaterialEvaluator);

        assert_eq!(position, before);
    }

    #[test]
    fn state_is_restored_even_from_a_tactical_position() {
        // Kiwipete: dense with captures, so alpha-beta cutoffs happen
        // on many different branches, exercising more of the
        // make/unmake pairing than a quiet position would.
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut position = Position::from_fen(fen).expect("valid FEN");
        let before = position.clone();

        search(&mut position, 3, &MaterialEvaluator);

        assert_eq!(position, before);
    }

    #[test]
    fn takes_a_free_undefended_queen() {
        // White rook can capture a black queen on a5 (same file) that
        // nothing defends; at a couple of plies deep this should be
        // the clear best move under pure material evaluation.
        let mut position =
            Position::from_fen("4k3/8/8/q7/8/8/8/R3K3 w - - 0 1").expect("valid FEN");

        let result = search(&mut position, 3, &MaterialEvaluator);

        let best_move = result.best_move.expect("should find a move");
        assert_eq!(best_move.from(), "a1".parse().unwrap());
        assert_eq!(best_move.to(), "a5".parse().unwrap());
    }

    #[test]
    fn stalemate_scores_zero_with_no_move() {
        // Classic stalemate: black king h8, white queen f7 and king g6
        // cover every square around it with no legal move and no
        // check.
        let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").expect("valid FEN");
        assert!(
            position.generate_legal_moves().is_empty(),
            "test setup: expected stalemate"
        );
        assert!(!position.in_check(), "test setup: stalemate, not checkmate");

        let mut position = position;
        let result = search(&mut position, 3, &MaterialEvaluator);

        assert_eq!(result.best_move, None);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn checkmate_at_root_reports_mate_score_and_no_move() {
        // White king h1 boxed in by its own pawns on f2/g2/h2, black
        // rook a1 delivering back-rank mate.
        let mut position =
            Position::from_fen("6k1/8/8/8/8/8/5PPP/r6K w - - 0 1").expect("valid FEN");
        assert!(
            position.generate_legal_moves().is_empty(),
            "test setup: expected checkmate"
        );
        assert!(
            position.in_check(),
            "test setup: expected checkmate, not stalemate"
        );

        let result = search(&mut position, 3, &MaterialEvaluator);

        assert_eq!(result.best_move, None);
        assert_eq!(mate_in_plies(result.score), Some(0));
    }

    #[test]
    fn prefers_shorter_mate_over_longer_mate() {
        // A position with a forced mate in 1 available alongside other
        // legal (non-mating) moves: search must choose the mate.
        let mut position =
            Position::from_fen("6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1").expect("valid FEN");

        let result = search(&mut position, 4, &MaterialEvaluator);

        assert_eq!(mate_in_plies(result.score), Some(1));
    }
}
