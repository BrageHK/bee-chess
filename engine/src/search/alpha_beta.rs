//! Negamax alpha-beta search, fixed-depth and time-bounded iterative
//! deepening: the first two slices of #6 (6a, plus 6c's iterative
//! deepening pulled forward to be time- rather than depth-driven).
//!
//! Deliberately excludes (see later #6 PRs): quiescence search (so it
//! suffers the horizon effect at the search boundary), a transposition
//! table, move ordering, aspiration windows, and PVS. None of those
//! affect correctness; they only affect speed and (for quiescence)
//! tactical horizon quality. Get the tree itself correct first.
//!
//! There is also no threading/cancellation infrastructure yet (that's
//! #7's territory) -- time-bounded search instead polls a `Deadline`
//! periodically from inside negamax and unwinds early when it's
//! passed. A partially-searched depth is discarded rather than
//! reported: alpha-beta's cutoffs assume a subtree was fully explored,
//! so a score produced after bailing out partway through one is not
//! trustworthy the way a fully-completed depth's score is.

use crate::chess::{Move, Position};
use crate::eval::Evaluator;

use super::deadline::Deadline;
use super::{Score, SearchResult, SCORE_INF, SCORE_MATE};

/// Searches `position` to exactly `depth` plies using negamax
/// alpha-beta, scoring leaves with `evaluator`. Returns the best move
/// found at the root (or `None` if the root position is checkmate or
/// stalemate), the score and principal variation of that line from
/// the root side to move's perspective, and the total number of nodes
/// visited.
///
/// `position` is restored to its exact starting state before this
/// returns: every recursive `make_move` is paired with an `unmake_move`
/// on every path, including the ones alpha-beta cuts off early.
///
/// This is fixed-depth: it always runs to completion at `depth`, with
/// no time limit. For time-bounded search that reports progress after
/// each completed depth, see `search_iterative`.
#[must_use]
pub fn search(position: &mut Position, depth: u32, evaluator: &impl Evaluator) -> SearchResult {
    // A fixed-depth search never times out: same code path as
    // search_iterative's per-depth search, just with an unlimited
    // deadline, so a single implementation serves both.
    search_to_depth(position, depth, evaluator, &Deadline::none())
        .expect("Deadline::none() never expires, so this can't be an incomplete search")
}

/// Searches `position` with iterative deepening (depth 1, then 2, then
/// 3, ...), stopping once `budget` has elapsed, and calling
/// `on_depth_complete` after each depth that finishes within budget.
/// Returns the result of the last depth that completed in time -- a
/// depth that was cut off partway through by the deadline is never
/// reported, since its score can't be trusted (see the module docs).
///
/// Always completes at least depth 1 regardless of `budget`, even a
/// zero or already-elapsed one: a legal `bestmove` must be returned
/// somehow, and depth 1 (one ply of the game's real legal moves,
/// scored by material) is a good enough floor for that -- there's no
/// cancellation machinery yet to bail out of an in-progress depth 1
/// and still have anything sensible to report.
pub fn search_iterative(
    position: &mut Position,
    budget: std::time::Duration,
    evaluator: &impl Evaluator,
    mut on_depth_complete: impl FnMut(&SearchResult),
) -> SearchResult {
    let deadline = Deadline::from_now(budget);

    let mut depth = 1;
    let mut last_completed = search_to_depth(position, depth, evaluator, &Deadline::none())
        .expect("depth 1 always completes: Deadline::none() never expires");
    on_depth_complete(&last_completed);

    // If depth 1 already found a forced mate, searching deeper cannot
    // improve on "I have found a way to win," and every ply deeper is
    // meaningfully more expensive -- stop immediately rather than
    // burning the rest of the time budget for no gain.
    if super::mate_in_plies(last_completed.score).is_some() {
        return last_completed;
    }

    loop {
        depth += 1;
        match search_to_depth(position, depth, evaluator, &deadline) {
            Some(result) => {
                let found_mate = super::mate_in_plies(result.score).is_some();
                last_completed = result;
                on_depth_complete(&last_completed);
                if found_mate {
                    return last_completed;
                }
            }
            None => return last_completed, // this depth was cut off; keep the previous one
        }

        if deadline.is_expired(0) {
            // is_expired(0) forces an actual clock check regardless of
            // node-count parity, since we're asking between depths,
            // not from inside the hot loop.
            return last_completed;
        }
    }
}

/// Searches to exactly `depth`, or returns `None` if `deadline` expired
/// partway through (in which case `position` is still fully restored,
/// but the search is incomplete and must be discarded by the caller).
fn search_to_depth(
    position: &mut Position,
    depth: u32,
    evaluator: &impl Evaluator,
    deadline: &Deadline,
) -> Option<SearchResult> {
    let mut nodes = 0u64;
    let moves = position.generate_legal_moves();

    if moves.is_empty() {
        // Root is checkmate or stalemate: nothing to play, but still a
        // well-defined score.
        let score = terminal_score(position, 0);
        return Some(SearchResult {
            best_move: None,
            score,
            nodes: 1,
            depth,
            pv: Vec::new(),
        });
    }

    let mut best_move = moves[0];
    let mut best_score = -SCORE_INF;
    let mut best_pv: Vec<Move> = Vec::new();
    let mut alpha = -SCORE_INF;
    let beta = SCORE_INF;

    for mv in moves {
        let undo = position.make_move(mv);
        let outcome = negamax(
            position,
            depth - 1,
            -beta,
            -alpha,
            1,
            evaluator,
            &mut nodes,
            deadline,
        );
        position.unmake_move(mv, undo);

        let Some((score, mut child_pv)) = outcome.map(|(s, pv)| (-s, pv)) else {
            return None; // ran out of time partway through the root move loop
        };

        if score > best_score {
            best_score = score;
            best_move = mv;
            child_pv.insert(0, mv);
            best_pv = child_pv;
        }
        alpha = alpha.max(score);
        // No beta cutoff at the root: we need to have actually
        // compared every move to know which one is best, not just
        // that some move is "good enough."
    }

    Some(SearchResult {
        best_move: Some(best_move),
        score: best_score,
        nodes: nodes + 1, // +1 for the root position itself
        depth,
        pv: best_pv,
    })
}

/// The recursive negamax search. `ply` is the distance from the root,
/// used only to ply-adjust mate scores (see `SCORE_MATE`'s docs) so a
/// shorter forced mate is always preferred over a longer one and a
/// losing side delays mate as long as possible.
///
/// Returns `None` if `deadline` expired during this call or any of its
/// children -- the caller must treat that as "this subtree's result is
/// unusable," not as a real (if pessimistic) score.
///
/// Returns the score together with the remaining principal variation
/// below this node (not including this node's own move -- the caller
/// prepends that). This allocates a small `Vec` per call, which is not
/// how a fast engine ultimately wants to collect a PV; correctness and
/// simplicity come first here, per the same reasoning as the rest of
/// this milestone -- see the module docs.
#[allow(clippy::too_many_arguments)]
fn negamax(
    position: &mut Position,
    depth: u32,
    mut alpha: Score,
    beta: Score,
    ply: u32,
    evaluator: &impl Evaluator,
    nodes: &mut u64,
    deadline: &Deadline,
) -> Option<(Score, Vec<Move>)> {
    if deadline.is_expired(*nodes) {
        return None;
    }

    *nodes += 1;

    let moves = position.generate_legal_moves();
    if moves.is_empty() {
        return Some((terminal_score(position, ply), Vec::new()));
    }

    if depth == 0 {
        return Some((evaluator.evaluate(position), Vec::new()));
    }

    let mut best = -SCORE_INF;
    let mut best_pv: Vec<Move> = Vec::new();

    for mv in moves {
        let undo = position.make_move(mv);
        let outcome = negamax(
            position,
            depth - 1,
            -beta,
            -alpha,
            ply + 1,
            evaluator,
            nodes,
            deadline,
        );
        position.unmake_move(mv, undo);

        let (score, mut child_pv) = match outcome {
            Some((s, pv)) => (-s, pv),
            None => return None,
        };

        if score > best {
            best = score;
            child_pv.insert(0, mv);
            best_pv = child_pv;
        }
        alpha = alpha.max(score);

        if alpha >= beta {
            break; // beta cutoff: the opponent won't allow this line
        }
    }

    Some((best, best_pv))
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
    use std::time::Duration;

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
        assert_eq!(result.pv.first(), Some(&best_move));
    }

    #[test]
    fn state_is_fully_restored_after_search() {
        let mut position = Position::startpos();
        let before = position.clone();

        let _ = search(&mut position, 4, &MaterialEvaluator);

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

        let _ = search(&mut position, 3, &MaterialEvaluator);

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

    #[test]
    fn iterative_deepening_reports_increasing_depths() {
        let mut position = Position::startpos();
        let mut depths_seen = Vec::new();

        search_iterative(
            &mut position,
            Duration::from_millis(200),
            &MaterialEvaluator,
            |result| depths_seen.push(result.depth),
        );

        assert!(
            depths_seen.len() >= 2,
            "should complete more than one depth in 200ms"
        );
        assert_eq!(depths_seen, {
            let mut sorted = depths_seen.clone();
            sorted.sort_unstable();
            sorted
        });
        assert_eq!(depths_seen.first(), Some(&1));
        // Strictly increasing, no repeats or gaps backward.
        for pair in depths_seen.windows(2) {
            assert_eq!(pair[1], pair[0] + 1);
        }
    }

    #[test]
    fn iterative_deepening_always_completes_at_least_depth_one() {
        let mut position = Position::startpos();

        // A zero budget: depth 1 must still complete and be returned,
        // since there's no cancellation machinery to bail out of an
        // in-progress depth 1 and still have a legal move to report.
        let result = search_iterative(
            &mut position,
            Duration::from_millis(0),
            &MaterialEvaluator,
            |_| {},
        );

        assert!(result.best_move.is_some());
        assert_eq!(result.depth, 1);
    }

    #[test]
    fn iterative_deepening_state_is_fully_restored() {
        let mut position = Position::startpos();
        let before = position.clone();

        search_iterative(
            &mut position,
            Duration::from_millis(50),
            &MaterialEvaluator,
            |_| {},
        );

        assert_eq!(position, before);
    }

    #[test]
    fn iterative_deepening_stops_immediately_once_mate_is_found() {
        // Mate in one is already found at depth 1 for this position
        // (there's exactly one legal move, and it delivers mate), so
        // a generous budget should still return almost immediately
        // rather than search deeper for no reason.
        let mut position =
            Position::from_fen("6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1").expect("valid FEN");
        let mut depths_seen = Vec::new();

        let result = search_iterative(
            &mut position,
            Duration::from_secs(5),
            &MaterialEvaluator,
            |r| depths_seen.push(r.depth),
        );

        assert_eq!(mate_in_plies(result.score), Some(1));
        // Should not have searched many depths once mate was found --
        // in particular, not anywhere close to filling a 5 second
        // budget's worth of deepening.
        assert!(depths_seen.len() <= 2, "depths searched: {depths_seen:?}");
    }

    #[test]
    fn iterative_deepening_pv_starts_with_the_best_move() {
        let mut position =
            Position::from_fen("6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1").expect("valid FEN");

        let result = search_iterative(
            &mut position,
            Duration::from_millis(100),
            &MaterialEvaluator,
            |_| {},
        );

        assert_eq!(result.pv.first(), result.best_move.as_ref());
    }
}
