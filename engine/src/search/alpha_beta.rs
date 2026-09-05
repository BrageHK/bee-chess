//! Negamax alpha-beta/PVS search with iterative deepening, quiescence,
//! move ordering (TT move, MVV-LVA, killers, and history), a bounded
//! transposition table, and repetition/fifty-move draw scoring.
//!
//! Iterative deepening deliberately keeps a full root window rather than
//! adding aspiration windows: PVS already supplies narrow windows below the
//! root, while a full root window avoids deadline-expensive fail-high/low
//! re-searches when the score changes sharply between depths.
//!
//! There is also no threading/cancellation infrastructure yet (that's
//! #7's territory) -- time-bounded search instead polls a `Deadline`
//! periodically from inside negamax and unwinds early when it's
//! passed. A partially-searched depth is discarded rather than
//! reported: alpha-beta's cutoffs assume a subtree was fully explored,
//! so a score produced after bailing out partway through one is not
//! trustworthy the way a fully-completed depth's score is.

use std::collections::HashMap;

use crate::chess::{Move, MoveFlag, PieceKind, Position};
use crate::eval::Evaluator;

use super::deadline::Deadline;
use super::{Score, SearchResult, SCORE_INF, SCORE_MATE};

const MAX_TT_ENTRIES: usize = 1 << 20;

#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    depth: u32,
    score: Score,
    bound: Bound,
    best_move: Option<Move>,
}

struct SearchState {
    table: HashMap<(u64, u32, u8), TtEntry>,
    killers: Vec<[Option<Move>; 2]>,
    history: [i32; 64 * 64],
    root_best: Option<Move>,
}

impl Default for SearchState {
    fn default() -> Self {
        Self {
            table: HashMap::new(),
            killers: Vec::new(),
            history: [0; 64 * 64],
            root_best: None,
        }
    }
}

fn normalized_history(position: &Position, history: &[u64]) -> Vec<u64> {
    let current = position.zobrist_hash();
    let mut path = history.to_vec();
    if path.last().copied() != Some(current) {
        path.push(current);
    }
    path
}

fn repetition_count(path: &[u64], hash: u64) -> u8 {
    path.iter()
        .filter(|&&seen| seen == hash)
        .count()
        .min(u8::MAX as usize) as u8
}

fn is_rule_draw(position: &Position, path: &[u64]) -> bool {
    position.halfmove_clock() >= 100 || repetition_count(path, position.zobrist_hash()) >= 3
}

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
    let history = [position.zobrist_hash()];
    search_with_history(position, depth, evaluator, &history)
}

/// Fixed-depth search with the hashes that led to `position`, used to score
/// threefold repetition inside the tree. The final hash should be the current
/// position; it is added defensively if the caller omits it.
pub fn search_with_history(
    position: &mut Position,
    depth: u32,
    evaluator: &impl Evaluator,
    history: &[u64],
) -> SearchResult {
    let mut state = SearchState::default();
    let mut path = normalized_history(position, history);
    // A fixed-depth search never times out: same code path as
    // search_iterative's per-depth search, just with an unlimited
    // deadline, so a single implementation serves both.
    search_to_depth(
        position,
        depth,
        evaluator,
        &Deadline::none(),
        &mut state,
        &mut path,
    )
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
    on_depth_complete: impl FnMut(&SearchResult),
) -> SearchResult {
    let history = [position.zobrist_hash()];
    search_iterative_with_history(position, budget, evaluator, &history, on_depth_complete)
}

pub fn search_iterative_with_history(
    position: &mut Position,
    budget: std::time::Duration,
    evaluator: &impl Evaluator,
    history: &[u64],
    mut on_depth_complete: impl FnMut(&SearchResult),
) -> SearchResult {
    let deadline = Deadline::from_now(budget);
    let mut state = SearchState::default();
    let mut path = normalized_history(position, history);

    let mut depth = 1;
    let mut last_completed = search_to_depth(
        position,
        depth,
        evaluator,
        &Deadline::none(),
        &mut state,
        &mut path,
    )
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
        match search_to_depth(position, depth, evaluator, &deadline, &mut state, &mut path) {
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
    state: &mut SearchState,
    path: &mut Vec<u64>,
) -> Option<SearchResult> {
    let mut nodes = 0u64;
    let mut moves = position.generate_legal_moves();

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

    // Checkmate/stalemate above take precedence; otherwise a claimable
    // repetition or fifty-move draw is an exact zero even though UCI still
    // needs a legal move to return.
    if is_rule_draw(position, path) {
        let best_move = moves[0];
        return Some(SearchResult {
            best_move: Some(best_move),
            score: 0,
            nodes: 1,
            depth,
            pv: vec![best_move],
        });
    }

    order_moves(position, &mut moves, state, 0, state.root_best);
    let mut best_move = moves[0];
    let mut best_score = -SCORE_INF;
    let mut best_pv: Vec<Move> = Vec::new();
    let mut alpha = -SCORE_INF;
    let beta = SCORE_INF;

    for mv in moves {
        let undo = position.make_move(mv);
        path.push(position.zobrist_hash());
        let outcome = negamax(
            position,
            depth - 1,
            -beta,
            -alpha,
            1,
            evaluator,
            &mut nodes,
            deadline,
            state,
            path,
        );
        path.pop();
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

    state.root_best = Some(best_move);
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
    mut beta: Score,
    ply: u32,
    evaluator: &impl Evaluator,
    nodes: &mut u64,
    deadline: &Deadline,
    state: &mut SearchState,
    path: &mut Vec<u64>,
) -> Option<(Score, Vec<Move>)> {
    if deadline.is_expired(*nodes) {
        return None;
    }

    *nodes += 1;

    if is_rule_draw(position, path) {
        return Some((0, Vec::new()));
    }

    let original_alpha = alpha;
    let original_beta = beta;
    let repetition = repetition_count(path, position.zobrist_hash());
    let tt_key = (
        position.zobrist_hash(),
        position.halfmove_clock(),
        repetition,
    );
    let tt_move = state.table.get(&tt_key).and_then(|entry| entry.best_move);
    if let Some(entry) = state
        .table
        .get(&tt_key)
        .copied()
        .filter(|entry| entry.depth >= depth)
    {
        let score = score_from_tt(entry.score, ply);
        match entry.bound {
            Bound::Exact => return Some((score, entry.best_move.into_iter().collect())),
            Bound::Lower => alpha = alpha.max(score),
            Bound::Upper => beta = beta.min(score),
        }
        if alpha >= beta {
            return Some((score, Vec::new()));
        }
    }

    let mut moves = position.generate_legal_moves();
    if moves.is_empty() {
        return Some((terminal_score(position, ply), Vec::new()));
    }

    if depth == 0 {
        let score = quiescence(
            position, alpha, beta, ply, ply, evaluator, nodes, deadline, path,
        )?;
        return Some((score, Vec::new()));
    }

    order_moves(position, &mut moves, state, ply as usize, tt_move);
    let mut best = -SCORE_INF;
    let mut best_pv: Vec<Move> = Vec::new();
    let mut best_move = None;

    for (move_index, mv) in moves.into_iter().enumerate() {
        let undo = position.make_move(mv);
        path.push(position.zobrist_hash());
        let mut outcome = if move_index == 0 {
            negamax(
                position,
                depth - 1,
                -beta,
                -alpha,
                ply + 1,
                evaluator,
                nodes,
                deadline,
                state,
                path,
            )
        } else {
            // Principal Variation Search: prove later moves fail low with a
            // null window, then re-search only an unexpected improvement.
            let scout = negamax(
                position,
                depth - 1,
                -alpha - 1,
                -alpha,
                ply + 1,
                evaluator,
                nodes,
                deadline,
                state,
                path,
            );
            match scout {
                Some((child_score, _)) if -child_score > alpha && -child_score < beta => negamax(
                    position,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    evaluator,
                    nodes,
                    deadline,
                    state,
                    path,
                ),
                other => other,
            }
        };
        path.pop();
        position.unmake_move(mv, undo);

        let (score, mut child_pv) = match outcome.take() {
            Some((s, pv)) => (-s, pv),
            None => return None,
        };

        if score > best {
            best = score;
            best_move = Some(mv);
            child_pv.insert(0, mv);
            best_pv = child_pv;
        }
        alpha = alpha.max(score);

        if alpha >= beta {
            record_cutoff(position, state, mv, ply as usize, depth);
            break; // beta cutoff: the opponent won't allow this line
        }
    }

    let bound = if best <= original_alpha {
        Bound::Upper
    } else if best >= original_beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    let should_replace = state
        .table
        .get(&tt_key)
        .is_none_or(|entry| depth >= entry.depth);
    if should_replace {
        if state.table.len() >= MAX_TT_ENTRIES {
            state.table.clear();
        }
        state.table.insert(
            tt_key,
            TtEntry {
                depth,
                score: score_to_tt(best, ply),
                bound,
                best_move,
            },
        );
    }

    Some((best, best_pv))
}

/// How many plies deep quiescence will keep searching captures below
/// the point it's called (`ply` at entry, not the root). Without this,
/// a queen-and-rook-dense middlegame with many possible captures per
/// side (e.g. the Kiwipete test position below) can take effectively
/// forever: every capture is searched regardless of whether it's a
/// good trade (no SEE/delta pruning here yet -- a real follow-up, not
/// this cap), so branching stays wide at every ply of the exchange, not
/// just deep. Measured against Kiwipete: quiescence's own node count
/// roughly 10x's per additional ply allowed here, so this needs to stay
/// small, not just finite -- a generous-looking cap (e.g. 16) still
/// lets a single leaf's quiescence run into the tens of millions of
/// nodes on a position like this. 4 plies (two full moves of exchange)
/// covers the overwhelming majority of real capture sequences (which
/// resolve via a short back-and-forth on one square) while keeping the
/// pathological wide-branching case bounded. Past the cap, quiescence
/// returns the stand-pat score instead of recursing further, the same
/// way `negamax` returns `evaluator`'s score at `depth == 0` -- a real
/// (if not fully exchange-resolved) evaluation, never an invented one.
const MAX_QUIESCENCE_PLY: u32 = 4;

/// Quiescence search: from `depth == 0`, keeps searching captures only
/// (a "noisy" position with hanging material can't be trusted just
/// because the depth budget ran out mid-exchange -- see the module
/// docs' "horizon effect" mention) until the position is "quiet"
/// (no more captures to consider) or `MAX_QUIESCENCE_PLY` is reached,
/// then returns `evaluator`'s static score for that position.
///
/// This is a stand-pat alpha-beta: unlike `negamax`, a leaf here isn't
/// forced to make a move at all. `evaluator.evaluate(position)` (the
/// "stand-pat score") is itself a candidate result -- the side to move
/// can always just decline every further capture -- so it seeds `best`
/// and `alpha` before any capture is tried, and a capture is only worth
/// recursing into if it can beat that baseline. This is what bounds the
/// search: without stand-pat, quiescence would have to prove a losing
/// capture is losing by searching it out, instead of pruning it
/// immediately for scoring worse than just not capturing.
///
/// `start_ply` is the ply this quiescence call tree was entered at (the
/// `ply` negamax was at when it hit `depth == 0`), used to measure
/// depth *within* quiescence against `MAX_QUIESCENCE_PLY` separately
/// from `ply`'s ordinary role of ply-adjusting mate scores.
///
/// Same `None`-means-deadline-expired contract as `negamax`.
#[allow(clippy::too_many_arguments)]
fn quiescence(
    position: &mut Position,
    mut alpha: Score,
    beta: Score,
    ply: u32,
    start_ply: u32,
    evaluator: &impl Evaluator,
    nodes: &mut u64,
    deadline: &Deadline,
    path: &mut Vec<u64>,
) -> Option<Score> {
    if deadline.is_expired(*nodes) {
        return None;
    }

    *nodes += 1;

    if is_rule_draw(position, path) {
        return Some(0);
    }

    // Checkmate/stalemate must still be detected even inside
    // quiescence: a position with no legal moves at all has no stand-pat
    // baseline to fall back on (there's no "declining every capture" if
    // there's no legal move whatsoever), so this needs the full legal
    // move list, not just captures, to tell those two cases apart from
    // an ordinary quiet position.
    let moves = position.generate_legal_moves();
    if moves.is_empty() {
        return Some(terminal_score(position, ply));
    }

    let stand_pat = evaluator.evaluate(position);
    if stand_pat >= beta {
        return Some(stand_pat); // opponent already wouldn't allow reaching this quiet line
    }
    let mut best = stand_pat;
    alpha = alpha.max(stand_pat);

    if ply - start_ply >= MAX_QUIESCENCE_PLY {
        return Some(best); // safety valve -- see MAX_QUIESCENCE_PLY's docs
    }

    let mut captures: Vec<Move> = moves
        .into_iter()
        .filter(|&mv| is_capture(position, mv))
        .collect();
    captures.sort_unstable_by_key(|&mv| std::cmp::Reverse(capture_order_score(position, mv)));
    for mv in captures {
        let undo = position.make_move(mv);
        path.push(position.zobrist_hash());
        let outcome = quiescence(
            position,
            -beta,
            -alpha,
            ply + 1,
            start_ply,
            evaluator,
            nodes,
            deadline,
            path,
        );
        path.pop();
        position.unmake_move(mv, undo);

        let score = -outcome?;

        if score > best {
            best = score;
        }
        alpha = alpha.max(score);

        if alpha >= beta {
            break; // beta cutoff, same reasoning as negamax's
        }
    }

    Some(best)
}

fn order_moves(
    position: &Position,
    moves: &mut [Move],
    state: &SearchState,
    ply: usize,
    tt_move: Option<Move>,
) {
    moves.sort_unstable_by_key(|&mv| {
        let score = if Some(mv) == tt_move {
            2_000_000
        } else if is_capture(position, mv) || mv.flag().promotion_kind().is_some() {
            1_000_000 + capture_order_score(position, mv)
        } else if state
            .killers
            .get(ply)
            .is_some_and(|killers| killers[0] == Some(mv))
        {
            900_000
        } else if state
            .killers
            .get(ply)
            .is_some_and(|killers| killers[1] == Some(mv))
        {
            800_000
        } else {
            state.history[history_index(mv)]
        };
        std::cmp::Reverse(score)
    });
}

fn record_cutoff(position: &Position, state: &mut SearchState, mv: Move, ply: usize, depth: u32) {
    if is_capture(position, mv) || mv.flag().promotion_kind().is_some() {
        return;
    }
    if state.killers.len() <= ply {
        state.killers.resize(ply + 1, [None; 2]);
    }
    if state.killers[ply][0] != Some(mv) {
        state.killers[ply][1] = state.killers[ply][0];
        state.killers[ply][0] = Some(mv);
    }
    let bonus = (depth * depth).min(i32::MAX as u32) as i32;
    state.history[history_index(mv)] = state.history[history_index(mv)].saturating_add(bonus);
}

const fn history_index(mv: Move) -> usize {
    mv.from().index() as usize * 64 + mv.to().index() as usize
}

fn capture_order_score(position: &Position, mv: Move) -> i32 {
    let attacker = position
        .piece_at(mv.from())
        .map_or(0, |piece| ordering_piece_value(piece.kind));
    let victim = if mv.flag() == MoveFlag::EnPassant {
        ordering_piece_value(PieceKind::Pawn)
    } else {
        position
            .piece_at(mv.to())
            .map_or(0, |piece| ordering_piece_value(piece.kind))
    };
    let promotion = mv.flag().promotion_kind().map_or(0, ordering_piece_value);
    victim * 16 - attacker + promotion
}

const fn ordering_piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight => 320,
        PieceKind::Bishop => 330,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 20_000,
    }
}

fn score_to_tt(score: Score, ply: u32) -> Score {
    if score >= SCORE_MATE - 1_000 {
        score + ply as Score
    } else if score <= -SCORE_MATE + 1_000 {
        score - ply as Score
    } else {
        score
    }
}

fn score_from_tt(score: Score, ply: u32) -> Score {
    if score >= SCORE_MATE - 1_000 {
        score - ply as Score
    } else if score <= -SCORE_MATE + 1_000 {
        score + ply as Score
    } else {
        score
    }
}

/// Whether `mv` captures a piece in `position`. Not carried on `Move`
/// itself (see `moves.rs`'s docs): a plain capture looks identical to a
/// quiet move without checking what's actually on the destination
/// square, except for en passant, whose destination square is always
/// empty (the captured pawn sits beside it, not on it).
fn is_capture(position: &Position, mv: Move) -> bool {
    mv.flag() == MoveFlag::EnPassant || position.piece_at(mv.to()).is_some()
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

    #[test]
    fn quiescence_resolves_a_hanging_capture_at_the_search_horizon() {
        // White rook on a1 can capture a black queen on a5 (same file,
        // nothing defends it) in one move. At depth 1, plain negamax
        // would stop right after that capture and score the position
        // by material alone -- which already sees the up-a-queen
        // material swing, since evaluation happens *after* the capturing
        // move is made. This isn't actually a horizon-effect case (the
        // gain is realized within the given depth either way); it exists
        // to confirm quiescence doesn't change a value that's already
        // correct at depth 1, i.e. it doesn't introduce a regression on
        // the simplest possible case before trusting it on subtler ones.
        let mut position =
            Position::from_fen("4k3/8/8/q7/8/8/8/R3K3 w - - 0 1").expect("valid FEN");

        let result = search(&mut position, 1, &MaterialEvaluator);

        let best_move = result.best_move.expect("should find a move");
        assert_eq!(best_move.from(), "a1".parse().unwrap());
        assert_eq!(best_move.to(), "a5".parse().unwrap());
    }

    #[test]
    fn quiescence_avoids_the_horizon_effect_of_a_losing_trade() {
        // White to move, depth 1: a rook on d4 can capture a black knight
        // on d5, but a black pawn on e6 recaptures it right back. Plain
        // depth-1 negamax (no quiescence) would stop immediately after
        // Rxd5 and score the position by material alone -- seeing only
        // "I won a knight" and missing that the rook falls right back
        // one ply later, the classic horizon effect. Quiescence must
        // keep searching this capture-recapture exchange past the
        // nominal depth-1 cutoff and correctly see the trade as a net
        // loss (rook for knight), so the engine should prefer leaving
        // its rook on d4 over grabbing the knight.
        let mut position =
            Position::from_fen("4k3/8/4p3/3n4/3R4/8/8/4K3 w - - 0 1").expect("valid FEN");

        let result = search(&mut position, 1, &MaterialEvaluator);

        let best_move = result.best_move.expect("should find a move");
        assert_ne!(
            (best_move.from(), best_move.to()),
            ("d4".parse().unwrap(), "d5".parse().unwrap()),
            "should not walk into a rook-for-knight trade that quiescence can see is losing"
        );
    }

    #[test]
    fn quiescence_never_makes_score_worse_than_stand_pat_when_no_capture_helps() {
        // A quiet position (no captures available at all for the side to
        // move) should score exactly the same at depth 1 as plain
        // material evaluation would -- quiescence must be a no-op here,
        // not perturb an already-quiet leaf's score.
        let mut position = Position::startpos();

        let result = search(&mut position, 1, &MaterialEvaluator);

        assert_eq!(result.score, MaterialEvaluator.evaluate(&position));
    }

    #[test]
    fn fifty_move_rule_scores_as_a_draw() {
        let mut position = Position::from_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 100 1").unwrap();
        let result = search(&mut position, 3, &MaterialEvaluator);
        assert_eq!(result.score, 0);
        assert!(
            result.best_move.is_some(),
            "UCI still requires a legal move"
        );
    }

    #[test]
    fn third_occurrence_scores_as_a_draw() {
        let mut position = Position::startpos();
        let hash = position.zobrist_hash();
        let result = search_with_history(&mut position, 3, &MaterialEvaluator, &[hash, hash, hash]);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn mvv_lva_orders_a_queen_capture_before_quiet_moves() {
        let position = Position::from_fen("4k3/8/8/q7/8/8/8/R3K3 w - - 0 1").unwrap();
        let mut moves = position.generate_legal_moves();
        order_moves(&position, &mut moves, &SearchState::default(), 0, None);
        assert_eq!(moves[0].from(), "a1".parse().unwrap());
        assert_eq!(moves[0].to(), "a5".parse().unwrap());
    }

    #[test]
    fn transposition_table_reuses_a_completed_search() {
        let mut position = Position::startpos();
        let mut state = SearchState::default();
        let mut path = vec![position.zobrist_hash()];
        let mut nodes = 0;
        let first = negamax(
            &mut position,
            3,
            -SCORE_INF,
            SCORE_INF,
            0,
            &MaterialEvaluator,
            &mut nodes,
            &Deadline::none(),
            &mut state,
            &mut path,
        )
        .unwrap();
        let first_nodes = nodes;
        let second = negamax(
            &mut position,
            3,
            -SCORE_INF,
            SCORE_INF,
            0,
            &MaterialEvaluator,
            &mut nodes,
            &Deadline::none(),
            &mut state,
            &mut path,
        )
        .unwrap();
        assert_eq!(second.0, first.0);
        assert_eq!(
            nodes - first_nodes,
            1,
            "the second search should hit the TT at its root"
        );
    }
}
