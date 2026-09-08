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

use crate::chess::{Color, Move, MoveFlag, Piece, PieceKind, Position, Square};
use crate::eval::Evaluator;

use super::deadline::{Deadline, StopSignal};
use super::{
    DeltaPruningStats, LmrStats, NullMoveStats, Score, SearchOptions, SearchResult, SeeStats,
    SCORE_INF, SCORE_MATE,
};

const MAX_TT_ENTRIES: usize = 1 << 20;
/// A practical ceiling for iterative deepening. Positions where every move
/// immediately reaches a rule draw can otherwise complete arbitrarily large
/// nominal depths in constant time, producing meaningless values in UCI
/// telemetry and experiment aggregates.
const MAX_ITERATIVE_DEPTH: u32 = 128;

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
    options: SearchOptions,
    table: HashMap<(u64, u32, u8), TtEntry>,
    killers: Vec<[Option<Move>; 2]>,
    history: [i32; 64 * 64],
    root_best: Option<Move>,
    lmr: LmrStats,
    null_move: NullMoveStats,
    delta_pruning: DeltaPruningStats,
    see_pruning: SeeStats,
}

impl SearchState {
    fn new(options: SearchOptions) -> Self {
        Self {
            options,
            table: HashMap::new(),
            killers: Vec::new(),
            history: [0; 64 * 64],
            root_best: None,
            lmr: LmrStats::default(),
            null_move: NullMoveStats::default(),
            delta_pruning: DeltaPruningStats::default(),
            see_pruning: SeeStats::default(),
        }
    }
}

impl Default for SearchState {
    fn default() -> Self {
        Self::new(SearchOptions::default())
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
    search_with_options(
        position,
        depth,
        evaluator,
        history,
        SearchOptions::default(),
    )
}

/// Same as `search_with_history`, with explicit `SearchOptions` -- see
/// that type's docs. `Engine` uses this to honor the `UseTT`/
/// `UseQuiescence` UCI options; every other caller (including every
/// existing test) goes through `search`/`search_with_history` and gets
/// `SearchOptions::default()` (both features on), so this is purely
/// additive.
pub fn search_with_options(
    position: &mut Position,
    depth: u32,
    evaluator: &impl Evaluator,
    history: &[u64],
    options: SearchOptions,
) -> SearchResult {
    let mut state = SearchState::new(options);
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
/// `budget` is used as both the soft and hard limit (see
/// `search_iterative_with_budget`'s docs for the distinction) -- this
/// entry point exists for callers (mostly tests, and any caller that
/// genuinely has just one number, not a real `TimeBudget`) that don't
/// need the split.
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
    on_depth_complete: impl FnMut(&SearchResult),
) -> SearchResult {
    search_iterative_with_options(
        position,
        budget,
        evaluator,
        history,
        SearchOptions::default(),
        on_depth_complete,
    )
}

/// Same as `search_iterative_with_history`, with explicit
/// `SearchOptions` -- see `search_with_options`'s docs for why this
/// exists alongside the unchanged, default-options entry points.
/// `budget` is used as both the soft and hard limit -- see
/// `search_iterative_with_budget` for the real soft/hard split.
///
/// Unlike `search_iterative_with_budget`, depth 1 here always runs
/// against `Deadline::none()` and so always completes -- this entry
/// point predates the fallback-move safety net (`Engine` always having
/// a legal move in hand before calling into search), so it keeps its
/// original guarantee for existing callers/tests rather than ever
/// returning `None`.
pub fn search_iterative_with_options(
    position: &mut Position,
    budget: std::time::Duration,
    evaluator: &impl Evaluator,
    history: &[u64],
    options: SearchOptions,
    mut on_depth_complete: impl FnMut(&SearchResult),
) -> SearchResult {
    let deadline = Deadline::from_now(budget);
    let mut state = SearchState::new(options);
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

    if super::mate_in_plies(last_completed.score).is_some() {
        return last_completed;
    }

    loop {
        if depth >= MAX_ITERATIVE_DEPTH {
            return last_completed;
        }
        depth += 1;
        match search_to_depth(position, depth, evaluator, &deadline, &mut state, &mut path) {
            Some(result) => {
                let found_mate = super::mate_in_plies(result.score).is_some();
                let search_saturated = result.nodes == last_completed.nodes
                    && result.score == last_completed.score
                    && result.best_move == last_completed.best_move;
                last_completed = result;
                on_depth_complete(&last_completed);
                if found_mate || search_saturated {
                    return last_completed;
                }
            }
            None => return last_completed,
        }

        if deadline.is_expired(0) {
            return last_completed;
        }
    }
}

/// Same as `search_iterative_with_options`, but also honors `stop` --
/// see `StopSignal`'s docs. Unlike `search_iterative_with_options`,
/// depth 1 here is *not* given an unconditional `Deadline::none()`:
/// once external cancellation is possible at all, even depth 1 must be
/// abortable (an already-requested `stop` right as `go` starts, say),
/// so this returns `Option<SearchResult>` the same way
/// `search_iterative_with_budget` does -- `None` means not even depth
/// 1 completed, and the caller (`Engine`) is responsible for having a
/// fallback move ready, exactly as it already is for
/// `search_iterative_with_budget`.
pub fn search_iterative_with_stop(
    position: &mut Position,
    budget: std::time::Duration,
    evaluator: &impl Evaluator,
    history: &[u64],
    options: SearchOptions,
    stop: StopSignal,
    mut on_depth_complete: impl FnMut(&SearchResult),
) -> Option<SearchResult> {
    let deadline = Deadline::from_now(budget).with_stop_signal(stop);
    let mut state = SearchState::new(options);
    let mut path = normalized_history(position, history);

    let mut depth = 1;
    let mut last_completed =
        search_to_depth(position, depth, evaluator, &deadline, &mut state, &mut path)?;
    on_depth_complete(&last_completed);

    if super::mate_in_plies(last_completed.score).is_some() {
        return Some(last_completed);
    }

    loop {
        depth += 1;
        match search_to_depth(position, depth, evaluator, &deadline, &mut state, &mut path) {
            Some(result) => {
                let found_mate = super::mate_in_plies(result.score).is_some();
                last_completed = result;
                on_depth_complete(&last_completed);
                if found_mate {
                    return Some(last_completed);
                }
            }
            None => return Some(last_completed),
        }

        if deadline.is_expired(0) {
            return Some(last_completed);
        }
    }
}

/// Searches `position` with iterative deepening under a real soft/hard
/// [`super::TimeBudget`] (see that type's docs for what each half
/// means) -- the entry point `Engine` uses once it has a clock-aware
/// `TimeManager` allocation rather than a single number.
///
/// A cut-off depth's result is always discarded (see the module
/// docs), *including depth 1*: unlike earlier versions of this
/// function, depth 1 is no longer given an unconditional
/// `Deadline::none()` -- under an extreme time control (e.g. `go
/// wtime 12`) even one ply, one move at a time, is not guaranteed to
/// finish before the hard limit. The invariant this function now
/// upholds is only ever "returns *some* legal move if one exists, and
/// never intentionally crosses the hard limit" -- see `Engine`'s own
/// `fallback_move`, which is what makes that safe: `Engine` always has
/// a legal move in hand *before* calling this, so a hard-limit abort
/// partway through depth 1 (returning `None` as `last_completed`) is a
/// well-defined, handleable outcome rather than "no move to report."
///
/// `stop` (see `StopSignal`'s docs) is attached to both the soft and
/// hard deadlines, so an external UCI `stop` cancels this exactly like
/// crossing the hard limit does -- the caller sees the same `None`-on-
/// nothing-completed / `Some((..))`-otherwise contract either way,
/// with no separate "was this a stop or a timeout" signal needed at
/// this layer.
///
/// Returns the `SearchResult` alongside a
/// [`super::TimeManagementTelemetry`] record of how the budget was
/// actually spent getting there (see that type's own docs for what
/// each field means and why it's not just folded into `SearchResult`
/// itself) -- this is the one place that has all the information
/// needed to build one (per-depth wall-clock timing, whether the final
/// attempted depth was aborted, and the root best-move/score history
/// across completed depths), so it's assembled here rather than
/// reconstructed by a caller from a sequence of `on_depth_complete`
/// calls after the fact.
///
/// Once the soft deadline hasn't been reached yet, this also estimates
/// the next depth's likely cost (from the last two completed depths'
/// own timings, see [`super::estimate_next_depth_cost`]) and skips
/// starting it at all if that estimate suggests it plausibly can't
/// finish before the hard deadline. This never replaces the hard
/// deadline itself as the correctness backstop -- an iteration that
/// *is* started can still be aborted mid-flight exactly as it always
/// could; prediction only ever avoids starting a depth this search
/// wouldn't have finished anyway, per the real `bee-tm` telemetry
/// (~17% of searches hitting the hard deadline mid-iteration in a
/// 20-game 10+0.1 Fischer experiment) that motivated building this.
/// This used to be switchable behind a `TimePolicy` UCI option
/// (`Baseline` vs `Predictive`); once the A/B experiment confirmed the
/// predictive check was a strict improvement, it was made the only
/// behavior and the option was deprecated (still accepted, ignored --
/// see `crate::uci`'s `setoption` handling).
pub fn search_iterative_with_budget(
    position: &mut Position,
    budget: super::TimeBudget,
    evaluator: &impl Evaluator,
    history: &[u64],
    options: SearchOptions,
    stop: StopSignal,
    mut on_depth_complete: impl FnMut(&SearchResult),
) -> Option<(SearchResult, super::TimeManagementTelemetry)> {
    let search_start = std::time::Instant::now();
    let soft_deadline = Deadline::from_now(budget.soft).with_stop_signal(stop.clone());
    let hard_deadline = Deadline::from_now(budget.hard).with_stop_signal(stop);
    let mut state = SearchState::new(options);
    let mut path = normalized_history(position, history);

    let mut best_move_changes = 0u32;
    // Updated only when a depth actually *completes* (never on an
    // aborted one, which has no trustworthy score to compare against
    // -- see `search_to_depth`'s module-level contract) -- this is
    // exactly `TimeManagementTelemetry::score_delta_cp`'s definition:
    // the delta between the last two *completed* depths, `None` until
    // there have been two of them.
    let mut last_score_delta: Option<Score> = None;
    let mut aborted = std::time::Duration::ZERO;
    // The last two completed depths' own wall-clock durations, oldest
    // first -- exactly what `estimate_next_depth_cost` needs to
    // project the next depth's likely cost. `previous_depth_duration`
    // starts `None` (only depth 1 has completed so far) and gains a
    // real value once depth 2 completes.
    #[allow(unused_assignments)]
    let mut previous_depth_duration: Option<std::time::Duration> = None;
    let mut last_depth_duration: std::time::Duration;

    let mut depth = 1;
    let depth_1_start = std::time::Instant::now();
    let mut last_completed = search_to_depth(
        position,
        depth,
        evaluator,
        &hard_deadline,
        &mut state,
        &mut path,
    )?;
    last_depth_duration = depth_1_start.elapsed();
    // Depth 1 itself being cut off (`?` returns `None` above) means no
    // `SearchResult` was ever produced, so there's no `SearchResult`
    // to pair telemetry with either -- the caller (`Engine`) falls
    // back to its own pre-chosen legal move in that case and has no
    // use for a telemetry record describing a search that found
    // nothing.
    let mut previous_score = Some(last_completed.score);
    let mut previous_best_move = last_completed.best_move;
    on_depth_complete(&last_completed);

    // If depth 1 already found a forced mate, searching deeper cannot
    // improve on "I have found a way to win," and every ply deeper is
    // meaningfully more expensive -- stop immediately rather than
    // burning the rest of the time budget for no gain.
    if super::mate_in_plies(last_completed.score).is_some() {
        return Some((
            last_completed,
            telemetry(budget, depth, aborted, best_move_changes, None),
        ));
    }

    loop {
        if depth >= MAX_ITERATIVE_DEPTH {
            return Some((
                last_completed,
                telemetry(budget, depth, aborted, best_move_changes, last_score_delta),
            ));
        }
        depth += 1;
        let depth_start = std::time::Instant::now();
        match search_to_depth(
            position,
            depth,
            evaluator,
            &hard_deadline,
            &mut state,
            &mut path,
        ) {
            Some(result) => {
                let found_mate = super::mate_in_plies(result.score).is_some();
                let search_saturated = result.nodes == last_completed.nodes
                    && result.score == last_completed.score
                    && result.best_move == last_completed.best_move;
                last_score_delta = previous_score.map(|previous| result.score - previous);
                if result.best_move != previous_best_move {
                    best_move_changes += 1;
                }
                previous_score = Some(result.score);
                previous_best_move = result.best_move;
                last_completed = result;
                previous_depth_duration = Some(last_depth_duration);
                last_depth_duration = depth_start.elapsed();
                on_depth_complete(&last_completed);
                if found_mate || search_saturated {
                    return Some((
                        last_completed,
                        telemetry(budget, depth, aborted, best_move_changes, last_score_delta),
                    ));
                }
            }
            None => {
                // This depth was cut off by the hard limit (or an
                // external stop) -- its result is discarded (per
                // `search_to_depth`'s contract), but the wall-clock
                // time spent computing it wasn't free, and is exactly
                // what `TimeManagementTelemetry::aborted_ms` exists to
                // surface: a search that regularly burns hundreds of
                // milliseconds on a depth it then throws away is the
                // concrete signal that predicting "can the next depth
                // plausibly finish" before starting it is worth
                // building. `depth - 1` here since `depth` itself is
                // the one that got cut off, not completed.
                aborted = depth_start.elapsed();
                return Some((
                    last_completed,
                    telemetry(
                        budget,
                        depth - 1,
                        aborted,
                        best_move_changes,
                        last_score_delta,
                    ),
                ));
            }
        }

        if soft_deadline.is_expired(0) {
            // is_expired(0) forces an actual clock check regardless of
            // node-count parity, since we're asking between depths,
            // not from inside the hot loop. Only the soft deadline is
            // checked here -- crossing it just means "don't start
            // another depth," not "abort the one that just finished."
            return Some((
                last_completed,
                telemetry(budget, depth, aborted, best_move_changes, last_score_delta),
            ));
        }

        // Predict how long the next depth will take, and skip starting
        // it at all if it plausibly won't fit -- see this function's
        // docs.
        let estimated_next_ms = super::estimate_next_depth_cost(
            last_depth_duration.as_millis() as u64,
            previous_depth_duration.map(|d| d.as_millis() as u64),
        );
        let affordable = super::next_depth_is_affordable(
            search_start.elapsed().as_millis() as u64,
            estimated_next_ms,
            budget.hard.as_millis() as u64,
        );
        if !affordable {
            // The next depth probably can't finish before the hard
            // deadline anyway -- skip starting it at all, rather
            // than starting it and almost certainly needing the hard
            // deadline to abort it partway through. The hard deadline
            // remains the backstop for every depth that *is* started.
            return Some((
                last_completed,
                telemetry(budget, depth, aborted, best_move_changes, last_score_delta),
            ));
        }
    }
}

/// Assembles a [`super::TimeManagementTelemetry`] record --
/// `search_iterative_with_budget`'s one job besides searching. Kept as
/// its own tiny function purely so every `return` site above states
/// its telemetry the same way rather than repeating the same struct
/// literal five times.
fn telemetry(
    budget: super::TimeBudget,
    completed_depth: u32,
    aborted: std::time::Duration,
    best_move_changes: u32,
    score_delta_cp: Option<Score>,
) -> super::TimeManagementTelemetry {
    super::TimeManagementTelemetry {
        soft_ms: budget.soft.as_millis() as u64,
        hard_ms: budget.hard.as_millis() as u64,
        completed_depth,
        aborted_ms: aborted.as_millis() as u64,
        best_move_changes,
        score_delta_cp,
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
    state.lmr = LmrStats::default();
    state.null_move = NullMoveStats::default();
    state.delta_pruning = DeltaPruningStats::default();
    state.see_pruning = SeeStats::default();
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
            lmr: state.lmr,
            null_move: state.null_move,
            delta_pruning: state.delta_pruning,
            see_pruning: state.see_pruning,
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
            lmr: state.lmr,
            null_move: state.null_move,
            delta_pruning: state.delta_pruning,
            see_pruning: state.see_pruning,
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
            true,
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
        lmr: state.lmr,
        null_move: state.null_move,
        delta_pruning: state.delta_pruning,
        see_pruning: state.see_pruning,
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
    allow_null: bool,
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
    // `UseTT` off means never probing or storing (see the store below):
    // `tt_move` simply stays `None`, so move ordering falls back to
    // MVV-LVA/killers/history alone, exactly as if no entry had ever
    // been found.
    let tt_move = if state.options.use_tt {
        state.table.get(&tt_key).and_then(|entry| entry.best_move)
    } else {
        None
    };
    if state.options.use_tt {
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
    }

    let mut moves = position.generate_legal_moves();
    if moves.is_empty() {
        return Some((terminal_score(position, ply), Vec::new()));
    }

    if depth == 0 {
        let score = if state.options.use_quiescence {
            quiescence(
                position, alpha, beta, ply, ply, evaluator, nodes, deadline, path, state,
            )?
        } else {
            evaluator.evaluate(position)
        };
        return Some((score, Vec::new()));
    }

    let null_move_reduction = null_move_reduction(depth, state.options.use_adaptive_null_move);
    let can_try_null = state.options.use_null_move
        && allow_null
        && depth >= null_move_reduction + 2
        && !position.in_check()
        && beta.abs() < SCORE_MATE - 1000
        && has_non_pawn_material(position, position.side_to_move());
    if can_try_null {
        state.null_move.attempts += 1;
        let undo = position.make_null_move();
        let outcome = negamax(
            position,
            depth - 1 - null_move_reduction,
            -beta,
            -beta + 1,
            ply + 1,
            evaluator,
            nodes,
            deadline,
            state,
            path,
            false,
        );
        position.unmake_null_move(undo);
        let (child_score, _) = outcome?;
        if -child_score >= beta {
            state.null_move.cutoffs += 1;
            return Some((-child_score, Vec::new()));
        }
    }

    order_moves(position, &mut moves, state, ply as usize, tt_move);
    let mut best = -SCORE_INF;
    let mut best_pv: Vec<Move> = Vec::new();
    let mut best_move = None;
    let in_check = position.in_check();

    for (move_index, mv) in moves.into_iter().enumerate() {
        let quiet = !is_capture(position, mv) && mv.flag().promotion_kind().is_none();
        let is_killer = state
            .killers
            .get(ply as usize)
            .is_some_and(|killers| killers.contains(&Some(mv)));
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
                true,
            )
        } else {
            // Principal Variation Search: prove later moves fail low with a
            // null window. Late quiet moves get a conservative one-ply
            // reduction; any reduced result that challenges alpha is first
            // verified at full depth before it can affect the result.
            let use_reduction = state.options.use_lmr
                && depth >= 3
                && move_index >= 4
                && quiet
                && !is_killer
                && !in_check
                && !position.in_check();
            if use_reduction {
                state.lmr.attempts += 1;
            }
            let mut scout = negamax(
                position,
                if use_reduction { depth - 2 } else { depth - 1 },
                -alpha - 1,
                -alpha,
                ply + 1,
                evaluator,
                nodes,
                deadline,
                state,
                path,
                true,
            );
            if use_reduction
                && scout
                    .as_ref()
                    .is_some_and(|(child_score, _)| -*child_score > alpha)
            {
                state.lmr.researches += 1;
                scout = negamax(
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
                    true,
                );
            }
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
                    true,
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

    if state.options.use_tt {
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
const DELTA_MARGIN: Score = 200;

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
    state: &mut SearchState,
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

    let must_evade_check = state.options.use_enhanced_quiescence && position.in_check();
    let stand_pat = evaluator.evaluate(position);
    let mut best = if must_evade_check {
        -SCORE_INF
    } else {
        if stand_pat >= beta {
            return Some(stand_pat); // opponent already wouldn't allow reaching this quiet line
        }
        alpha = alpha.max(stand_pat);
        stand_pat
    };

    if ply - start_ply >= MAX_QUIESCENCE_PLY {
        return Some(stand_pat); // safety valve -- see MAX_QUIESCENCE_PLY's docs
    }

    let mut noisy_moves: Vec<Move> = moves
        .into_iter()
        .filter(|&mv| {
            must_evade_check
                || is_capture(position, mv)
                || (state.options.use_enhanced_quiescence && mv.flag().promotion_kind().is_some())
        })
        .collect();
    noisy_moves.sort_unstable_by_key(|&mv| {
        std::cmp::Reverse(capture_ordering_score(position, mv, state.options))
    });
    for mv in noisy_moves {
        if should_delta_prune(position, mv, stand_pat, alpha, must_evade_check, state)
            || should_see_prune(position, mv, must_evade_check, state)
        {
            continue;
        }
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
            state,
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

fn should_delta_prune(
    position: &Position,
    mv: Move,
    stand_pat: Score,
    alpha: Score,
    must_evade_check: bool,
    state: &mut SearchState,
) -> bool {
    if !state.options.use_delta_pruning
        || must_evade_check
        || !is_capture(position, mv)
        || mv.flag().promotion_kind().is_some()
        || alpha.abs() >= SCORE_MATE - 1_000
        || !has_major_material(position)
    {
        return false;
    }

    state.delta_pruning.attempts += 1;
    let captured_value = if mv.flag() == MoveFlag::EnPassant {
        ordering_piece_value(PieceKind::Pawn)
    } else {
        position
            .piece_at(mv.to())
            .map_or(0, |piece| ordering_piece_value(piece.kind))
    };
    let prune = stand_pat
        .saturating_add(captured_value)
        .saturating_add(DELTA_MARGIN)
        < alpha;
    if prune {
        state.delta_pruning.pruned += 1;
    }
    prune
}

/// Whether `mv` (already known to be a capture -- see `should_delta_
/// prune`'s docs on the same contract) can be skipped in quiescence
/// because SEE judges it a clear material loss. Unlike delta pruning
/// (a margin-based heuristic bounding the *best case* a capture could
/// possibly reach), this is exact: if the full simulated exchange on
/// the destination square nets negative material, no positional
/// compensation quiescence itself could ever discover changes that --
/// quiescence only ever explores captures/checks/promotions, so a
/// losing trade's actual downstream position is never evaluated here
/// regardless.
///
/// Not applied while in check (must_evade_check) or with mate-range
/// bounds in play, for the same reasons `should_delta_prune` excludes
/// those cases: a check response can't be pruned away by a material
/// heuristic, and mate-range alpha/beta values aren't ordinary
/// centipawn comparisons SEE's material-only result should be
/// compared against.
fn should_see_prune(
    position: &Position,
    mv: Move,
    must_evade_check: bool,
    state: &mut SearchState,
) -> bool {
    if !state.options.use_see
        || must_evade_check
        || !is_capture(position, mv)
        || mv.flag().promotion_kind().is_some()
    {
        return false;
    }

    state.see_pruning.attempts += 1;
    let prune = static_exchange_evaluation(position, mv) < 0;
    if prune {
        state.see_pruning.pruned += 1;
    }
    prune
}

fn has_major_material(position: &Position) -> bool {
    (0..Square::COUNT as u8).any(|index| {
        position
            .piece_at(Square::new(index))
            .is_some_and(|piece| matches!(piece.kind, PieceKind::Rook | PieceKind::Queen))
    })
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
            1_000_000 + capture_ordering_score(position, mv, state.options)
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

fn has_non_pawn_material(position: &Position, color: Color) -> bool {
    (0..Square::COUNT as u8).any(|index| {
        position.piece_at(Square::new(index)).is_some_and(|piece| {
            piece.color == color && !matches!(piece.kind, PieceKind::Pawn | PieceKind::King)
        })
    })
}

const fn null_move_reduction(depth: u32, adaptive: bool) -> u32 {
    if adaptive && depth >= 7 {
        3
    } else {
        2
    }
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

/// The score `order_moves`/quiescence's capture sort actually uses for
/// a capture: SEE's real exchange result when `use_see` is on (falling
/// back to plain `capture_order_score`'s MVV-LVA heuristic when it's
/// off, or for a non-capturing promotion, which SEE has nothing to say
/// about -- promotions still need *some* ordering score, and MVV-LVA's
/// existing `victim * 16 - attacker + promotion` term already handles
/// that case correctly). A caller must still ensure `mv` is actually a
/// capture-or-promotion before calling this; it doesn't check that
/// itself, matching `capture_order_score`'s own contract.
fn capture_ordering_score(position: &Position, mv: Move, options: SearchOptions) -> i32 {
    if options.use_see && is_capture(position, mv) {
        static_exchange_evaluation(position, mv)
    } else {
        capture_order_score(position, mv)
    }
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

/// Static Exchange Evaluation: simulates the likely capture sequence on
/// `mv`'s destination square (both sides always recapturing with their
/// cheapest available attacker, per `Position::least_valuable_attacker`)
/// and returns the net material result in centipawns, from the mover's
/// perspective -- positive means the exchange favors whoever plays
/// `mv`, negative means it doesn't.
///
/// This is a heuristic, not truth: it ignores checks, pins, discovered
/// attacks, and any positional consequence of the exchange (see this
/// module's docs on why quiescence and alpha-beta are what actually
/// resolve real tactics; SEE only ever informs move ordering and
/// pruning decisions about *which* captures are worth investigating
/// further). Not called for `mv`s that aren't captures at all --
/// `capture_order_score`'s callers already filter for that.
///
/// Runs on a cloned scratch board via `Position::set_piece` rather than
/// `make_move`/`unmake_move`: the simulation only cares about which
/// pieces occupy which squares as the exchange proceeds, never about
/// turn order, castling rights, en passant, or move counters, so a
/// direct piece-by-piece mutation is both simpler and cheaper than
/// threading a full move through the normal make/unmake machinery.
fn static_exchange_evaluation(position: &Position, mv: Move) -> i32 {
    let target = mv.to();
    let mover_color = match position.piece_at(mv.from()) {
        Some(piece) => piece.color,
        // A capture always has a piece on its own `from` square in any
        // position this is ever called against; falling back to "the
        // exchange is worthless" rather than panicking keeps this
        // total for a caller that somehow asks about a malformed move.
        None => return 0,
    };
    let Some(mut attacker_kind) = position.piece_at(mv.from()).map(|piece| piece.kind) else {
        return 0;
    };

    let mut board = position.clone();
    // The first capture's gain is fixed by the move itself (en passant
    // captures a pawn that isn't actually on the destination square,
    // exactly like `is_capture`/`capture_order_score` already special-
    // case), not re-derived from `least_valuable_attacker` below.
    let mut gains = vec![if mv.flag() == MoveFlag::EnPassant {
        ordering_piece_value(PieceKind::Pawn)
    } else {
        board
            .piece_at(target)
            .map_or(0, |piece| ordering_piece_value(piece.kind))
    }];
    board.set_piece(mv.from(), None);
    if mv.flag() == MoveFlag::EnPassant {
        // The captured pawn sits beside the destination square, not on
        // it -- see `is_capture`'s docs.
        let captured_pawn_rank = mv.from().rank();
        board.set_piece(
            Square::from_file_rank(target.file(), captured_pawn_rank),
            None,
        );
    }
    board.set_piece(target, Some(Piece::new(attacker_kind, mover_color)));

    let mut side_to_capture = mover_color.opposite();
    while let Some((attacker_square, kind)) = board.least_valuable_attacker(target, side_to_capture)
    {
        // Each new gain is "what this recapture wins" (the value of
        // whatever is currently sitting on `target`, i.e. the previous
        // attacker) minus whatever the previous exchange already
        // banked, negated -- the standard SEE swap-list recurrence:
        // a recapture is only worth playing if it doesn't leave the
        // position worse than simply not recapturing at all, which
        // `resolve_see_gains` (the final minimax-over-the-list step
        // below) accounts for regardless of how deep the list goes.
        let captured_value = ordering_piece_value(attacker_kind);
        gains.push(captured_value - *gains.last().expect("gains is never empty"));

        board.set_piece(attacker_square, None);
        board.set_piece(target, Some(Piece::new(kind, side_to_capture)));
        attacker_kind = kind;
        side_to_capture = side_to_capture.opposite();
    }

    resolve_see_gains(&gains)
}

/// Folds a SEE swap list (`gains[0]` is the value of the very first
/// piece captured; each `gains[d]` after that is `piece_value(the
/// attacker that just moved onto the target square at step d-1) -
/// gains[d-1]`, exactly as `static_exchange_evaluation` builds it) into
/// the single net result the *first* mover actually achieves, assuming
/// both sides play optimally -- i.e. a side only "recaptures" if doing
/// so doesn't leave them worse off than simply stopping the exchange
/// right there. This is the standard SEE fold (see the chess
/// programming wiki's "Static Exchange Evaluation" article for the
/// same recurrence under the same name): walking backward,
/// `gains[i] = -max(-gains[i], gains[i+1])` -- "the side to move at
/// step `i` either takes the (negated, since it's now their turn to
/// decide) result of continuing at `i+1`, or declines and banks
/// `gains[i]` outright, whichever is better for them."
fn resolve_see_gains(gains: &[i32]) -> i32 {
    let mut folded: Vec<i32> = gains.to_vec();
    for i in (0..folded.len().saturating_sub(1)).rev() {
        folded[i] = -(-folded[i]).max(folded[i + 1]);
    }
    folded.first().copied().unwrap_or(0)
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
    fn iterative_deepening_caps_nominal_depth_in_immediate_draw_trees() {
        // At halfmove 99 every legal non-pawn, non-capture move reaches the
        // fifty-move draw immediately. Searching depth 2 or 20,000 therefore
        // costs almost the same unless iterative deepening has a ceiling.
        let mut position =
            Position::from_fen("4k3/8/8/8/8/8/8/4K2N w - - 99 1").expect("valid FEN");
        let mut reported_depths = Vec::new();

        let result = search_iterative(
            &mut position,
            std::time::Duration::from_secs(1),
            &MaterialEvaluator,
            |result| reported_depths.push(result.depth),
        );

        assert_eq!(result.depth, 2);
        assert_eq!(reported_depths, vec![1, 2]);
    }

    #[test]
    fn budgeted_search_stops_at_the_soft_deadline_without_waiting_for_hard() {
        // A generous hard limit but a tiny soft limit: iterative
        // deepening should stop starting new depths once soft has
        // elapsed, long before hard would ever kick in.
        let mut position = Position::startpos();
        let history = [position.zobrist_hash()];
        let start = std::time::Instant::now();

        let result = search_iterative_with_budget(
            &mut position,
            super::super::TimeBudget {
                soft: Duration::from_millis(20),
                hard: Duration::from_secs(10),
            },
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
            StopSignal::new(),
            |_| {},
        );

        assert!(result.is_some());
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "should have stopped at the soft deadline, not run anywhere near the hard one"
        );
    }

    #[test]
    fn budgeted_search_returns_none_when_even_depth_1_cannot_complete_before_the_hard_limit() {
        // An already-expired hard deadline: depth 1 itself must be
        // abortable now (unlike the older, unconditionally-`Deadline::
        // none()` depth-1 guarantee) -- the caller (`Engine`) is
        // responsible for having a fallback move ready in this case.
        let mut position = Position::startpos();
        let history = [position.zobrist_hash()];

        let result = search_iterative_with_budget(
            &mut position,
            super::super::TimeBudget {
                soft: Duration::ZERO,
                hard: Duration::ZERO,
            },
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
            StopSignal::new(),
            |_| {},
        );

        // This is a timing-sensitive edge case (a zero deadline is
        // already expired, but the very first clock check inside
        // negamax happens after a small number of nodes -- see
        // `Deadline`'s docs), so depth 1 completing anyway is an
        // acceptable outcome; what matters is that a `None` result
        // here doesn't panic and is a well-defined "use the fallback"
        // signal.
        if let Some((completed, _telemetry)) = result {
            assert_eq!(completed.depth, 1);
        }
    }

    #[test]
    fn telemetry_records_the_allocated_budget_and_completed_depth() {
        let mut position = Position::startpos();
        let history = [position.zobrist_hash()];
        let budget = super::super::TimeBudget {
            soft: Duration::from_millis(500),
            hard: Duration::from_secs(2),
        };

        let (result, telemetry) = search_iterative_with_budget(
            &mut position,
            budget,
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
            StopSignal::new(),
            |_| {},
        )
        .expect("depth 1 should complete with a generous budget");

        assert_eq!(telemetry.soft_ms, 500);
        assert_eq!(telemetry.hard_ms, 2000);
        assert_eq!(telemetry.completed_depth, result.depth);
        // `aborted_ms` itself is deliberately not asserted here: with a
        // real wall-clock budget, whether the final iteration happens
        // to land exactly on the soft boundary or instead gets cut off
        // by the hard one partway through depends on real timing (and
        // is *expected* to vary with machine speed/load -- e.g. a
        // slower CI runner can easily make an iteration still be
        // running when the hard deadline arrives, which is completely
        // normal, not a bug). See
        // `telemetry_reports_the_last_completed_depth_when_a_later_one_is_cancelled`
        // for a deterministic (StopSignal-driven, not timing-based)
        // test of the aborted-iteration path itself.
    }

    #[test]
    fn search_declines_to_start_a_depth_it_estimates_wont_finish() {
        // A budget where depth 1 and depth 2 both complete comfortably
        // (a generous soft/hard budget overall), but the hard budget
        // is deliberately tightened to just barely more than what
        // depth 1 + depth 2 together are expected to cost, on a
        // position deep enough that depth 3 is genuinely, measurably
        // more expensive than depth 2 (real branching-factor growth,
        // not a contrived StopSignal) -- search should stop *without*
        // attempting (and therefore without aborting) depth 3, rather
        // than starting it and only finding out it didn't fit by
        // actually trying it.
        //
        // This is inherently a real-timing-based test (unlike the
        // StopSignal-driven determinism used elsewhere in this file),
        // so it only asserts the one thing that's true regardless of
        // exact machine speed: whenever search stops *without* having
        // aborted an iteration (`aborted_ms == 0`), the depth it
        // stopped at must be the one it estimated, not one attempt-
        // and-abort would have reached instead -- i.e. this never
        // *aborts* a depth under this budget (soft is generous; only
        // the predictive check or the hard limit can stop it, and
        // hitting the hard limit exactly as a real depth boundary
        // lines up is possible but doesn't invalidate the claim).
        let mut position = Position::startpos();
        let history = [position.zobrist_hash()];
        let budget = super::super::TimeBudget {
            soft: Duration::from_secs(60), // never the limiting factor here
            hard: Duration::from_millis(150),
        };

        let (_result, telemetry) = search_iterative_with_budget(
            &mut position,
            budget,
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
            StopSignal::new(),
            |_| {},
        )
        .expect("depth 1 should always complete");

        // The core claim: a depth that genuinely wouldn't have fit is
        // never *started* (and therefore never needs aborting) once
        // there's real growth-factor history to estimate from --
        // `aborted_ms` should stay 0 far more reliably than it would
        // under the old start-whenever-the-soft-deadline-allows
        // behavior with the same tight hard budget (see the module's
        // own real-experiment telemetry showing that older behavior
        // aborting ~17% of searches under a comparable real hard
        // budget -- the evidence that motivated this check).
        assert_eq!(
            telemetry.aborted_ms, 0,
            "search should decline to start a depth it estimates won't fit, \
             rather than starting it and needing the hard deadline to abort it"
        );
    }

    #[test]
    fn telemetry_has_no_score_delta_when_only_depth_1_completes() {
        // Request cancellation right after depth 1 completes -- same
        // deterministic technique as
        // `telemetry_reports_the_last_completed_depth_when_a_later_one_is_cancelled`,
        // guaranteeing exactly one completed depth (never a second),
        // so there's nothing to compute a score delta or a best-move
        // change from yet.
        let mut position = Position::startpos();
        let history = [position.zobrist_hash()];
        let budget = super::super::TimeBudget {
            soft: Duration::from_secs(60),
            hard: Duration::from_secs(60),
        };
        let stop = StopSignal::new();
        let stop_from_callback = stop.clone();

        let (_result, telemetry) = search_iterative_with_budget(
            &mut position,
            budget,
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
            stop,
            move |_| stop_from_callback.request_stop(),
        )
        .expect("depth 1 should still complete");

        assert_eq!(telemetry.completed_depth, 1);
        assert_eq!(telemetry.score_delta_cp, None);
        assert_eq!(telemetry.best_move_changes, 0);
    }

    #[test]
    fn telemetry_counts_best_move_changes_across_completed_depths() {
        // A generous budget: iterative deepening should comfortably
        // reach several depths from the start position, giving
        // multiple completed-depth transitions to count changes across.
        let mut position = Position::startpos();
        let history = [position.zobrist_hash()];
        let budget = super::super::TimeBudget {
            soft: Duration::from_secs(1),
            hard: Duration::from_secs(3),
        };

        let mut best_moves = Vec::new();
        let (_result, telemetry) = search_iterative_with_budget(
            &mut position,
            budget,
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
            StopSignal::new(),
            |result| best_moves.push(result.best_move),
        )
        .expect("should complete at least depth 1");

        let actual_changes = best_moves
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count() as u32;
        assert_eq!(
            telemetry.best_move_changes, actual_changes,
            "telemetry's count must match the actual best-move transitions on_depth_complete observed"
        );
    }

    #[test]
    fn telemetry_reports_the_last_completed_depth_when_a_later_one_is_cancelled() {
        // Request cancellation as soon as depth 2 starts (from
        // `on_depth_complete`, called after depth 1) -- deterministic,
        // unlike relying on a hard-deadline race against real wall-
        // clock timing: depth 2 is guaranteed to be cut off by the
        // `StopSignal` (checked unconditionally on every node -- see
        // `Deadline::is_expired`'s docs), never completing, so
        // telemetry must report `completed_depth: 1`, not 2 -- even
        // though a depth-2 attempt genuinely started and was aborted.
        // (`aborted_ms` itself isn't asserted here: cancelling a
        // trivial depth-2 search from the start position typically
        // takes microseconds, which rounds down to 0 at
        // `Duration::as_millis`'s millisecond granularity -- see
        // `telemetry_records_the_allocated_budget_and_completed_depth`
        // for `aborted_ms`'s zero-when-nothing-aborted case, and the
        // module's own real-position search timings for evidence
        // `aborted_ms` is wired to something real when a cut-off
        // iteration actually takes measurable wall-clock time.)
        let mut position = Position::startpos();
        let history = [position.zobrist_hash()];
        let budget = super::super::TimeBudget {
            soft: Duration::from_secs(60),
            hard: Duration::from_secs(60),
        };
        let stop = StopSignal::new();
        let stop_from_callback = stop.clone();

        let (result, telemetry) = search_iterative_with_budget(
            &mut position,
            budget,
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
            stop,
            move |_| stop_from_callback.request_stop(),
        )
        .expect("depth 1 should complete before the stop takes effect");

        assert_eq!(
            result.depth, 1,
            "depth 2 must have been cancelled, not completed"
        );
        assert_eq!(telemetry.completed_depth, 1);
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
    fn resolve_see_gains_returns_the_single_gain_of_an_undefended_capture() {
        // Just "I capture a rook and nobody recaptures" -- the whole
        // exchange's value is exactly what the first capture won.
        assert_eq!(resolve_see_gains(&[500]), 500);
    }

    #[test]
    fn resolve_see_gains_takes_a_recapture_that_denies_more_than_it_costs() {
        // gains[0] = 500: the first capture wins a rook. gains[1] =
        // -180 encodes "if the opponent recaptures the (knight-valued)
        // attacker now sitting on the square, that costs the *first*
        // side's running total 320 - 500 = -180" (i.e. the opponent
        // gives up their knight but the first side is left with only
        // 500 - (320 knight lost in trade for what they captured) --
        // the standard SEE fold, `gains[0] = -max(-gains[0],
        // gains[1])`, correctly has the opponent *take* this recapture:
        // it reduces what the first side nets from 500 down to 180,
        // which is a better outcome for the opponent than letting the
        // full 500 stand, even though it costs them their knight.
        assert_eq!(resolve_see_gains(&[500, -180]), 180);
    }

    #[test]
    fn resolve_see_gains_walks_a_four_deep_swap_list_correctly() {
        // A hand-verified swap list, folded with the standard backward
        // recurrence `gains[i] = -max(-gains[i], gains[i+1])`, walking
        // i from len-2 down to 0:
        //   i=2: gains[2] = -max(-30, -10) = 10   -> [100, -20, 10, -10]
        //   i=1: gains[1] = -max(20, 10)   = -20  -> [100, -20, 10, -10]
        //   i=0: gains[0] = -max(-100, -20) = 20  -> [20, -20, 10, -10]
        assert_eq!(resolve_see_gains(&[100, -20, 30, -10]), 20);
    }

    fn see_move(position: &Position, from: &str, to: &str) -> Move {
        let from: Square = from.parse().unwrap();
        let to: Square = to.parse().unwrap();
        position
            .generate_legal_moves()
            .into_iter()
            .find(|mv| mv.from() == from && mv.to() == to)
            .unwrap_or_else(|| panic!("no legal move {from}{to} in {}", position.to_fen()))
    }

    #[test]
    fn see_scores_an_undefended_capture_as_a_clean_win() {
        // White rook takes an undefended pawn: nobody can recapture,
        // so the result is exactly the pawn's value.
        let position = Position::from_fen("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1").unwrap();
        let mv = see_move(&position, "a1", "a2");
        assert_eq!(static_exchange_evaluation(&position, mv), 100);
    }

    #[test]
    fn see_scores_a_defended_capture_as_a_loss_for_a_more_valuable_attacker() {
        // White queen takes a pawn on b2 that's defended by a knight on
        // d3 (a real knight move, d3-b2): the knight recaptures the
        // queen, so the exchange is a clear loss for White
        // (100 - 900 = -800) despite winning a pawn up front. White's
        // own king sits on h1 rather than e1, since d3's knight also
        // attacks e1 -- putting the king there would make every non-
        // king move illegal (already in check) rather than exercising
        // the capture this test is actually about.
        let position = Position::from_fen("4k3/8/8/8/8/3n4/1p6/Q6K w - - 0 1").unwrap();
        let mv = see_move(&position, "a1", "b2");
        assert_eq!(static_exchange_evaluation(&position, mv), 100 - 900);
    }

    #[test]
    fn see_recognizes_an_undefended_pawn_capture_by_a_minor_piece_as_favorable() {
        // White knight on c3 takes an undefended pawn on d1 (a real
        // knight move -- c3 to d1) -- a clean win of exactly the
        // pawn's value, with no recapture available at all.
        let position = Position::from_fen("4k3/8/8/8/8/2N5/8/3pK3 w - - 0 1").unwrap();
        let mv = see_move(&position, "c3", "d1");
        assert_eq!(static_exchange_evaluation(&position, mv), 100);
    }

    #[test]
    fn see_handles_a_multi_piece_exchange_correctly() {
        // White bishop on h2 takes a pawn on e5 (clean diagonal,
        // h2-g3-f4-e5), defended by a Black knight on d3 (a real
        // knight move, d3-e5). Nothing defends the knight in turn, so
        // the full sequence is: Bxe5 (+100), Nxe5 (-330 running) --
        // White should decline any further recapture (there isn't a
        // legal one here anyway), so the net result is a losing trade
        // for White: a bishop for a pawn (100 - 330 = -230). White's
        // king sits on h1, not e1 -- d3's knight also attacks e1 (see
        // the queen/knight test above for the same pitfall).
        let position = Position::from_fen("4k3/8/8/4p3/8/3n4/7B/7K w - - 0 1").unwrap();
        let mv = see_move(&position, "h2", "e5");
        assert_eq!(static_exchange_evaluation(&position, mv), 100 - 330);
    }

    #[test]
    fn see_handles_en_passant_captures() {
        // White pawn captures en passant on d6, winning the pawn on d5
        // (which does not sit on the destination square) -- nothing
        // else attacks d6, so the result is exactly one pawn.
        let position = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let mv = see_move(&position, "e5", "d6");
        assert_eq!(mv.flag(), MoveFlag::EnPassant);
        assert_eq!(static_exchange_evaluation(&position, mv), 100);
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
            true,
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
            true,
        )
        .unwrap();
        assert_eq!(second.0, first.0);
        assert_eq!(
            nodes - first_nodes,
            1,
            "the second search should hit the TT at its root"
        );
    }

    #[test]
    fn use_tt_false_disables_transposition_table_reuse() {
        // Same setup as `transposition_table_reuses_a_completed_search`,
        // but with `UseTT` off: repeating the identical search must not
        // short-circuit at the root the way a TT hit would, since
        // nothing was ever stored for this position.
        let mut position = Position::startpos();
        let mut state = SearchState::new(SearchOptions {
            use_tt: false,
            use_quiescence: true,
            use_enhanced_quiescence: true,
            use_lmr: true,
            use_null_move: true,
            use_adaptive_null_move: true,
            use_delta_pruning: true,
            use_see: true,
        });
        let mut path = vec![position.zobrist_hash()];
        let mut nodes = 0;
        negamax(
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
            true,
        )
        .unwrap();
        let first_nodes = nodes;
        negamax(
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
            true,
        )
        .unwrap();
        assert!(
            nodes - first_nodes > 1,
            "with UseTT off, repeating the search must redo the full node count, not hit a cached root"
        );
    }

    #[test]
    fn late_move_reductions_search_fewer_nodes() {
        let original = Position::startpos();
        let history = [original.zobrist_hash()];
        let mut with_lmr = original.clone();
        let mut without_lmr = original.clone();

        let reduced = search_with_options(
            &mut with_lmr,
            4,
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
        );
        let full = search_with_options(
            &mut without_lmr,
            4,
            &MaterialEvaluator,
            &history,
            SearchOptions {
                use_lmr: false,
                ..SearchOptions::default()
            },
        );

        assert!(
            reduced.nodes < full.nodes,
            "LMR should reduce work: {} vs {} nodes",
            reduced.nodes,
            full.nodes
        );
        assert_eq!(with_lmr, original);
        assert_eq!(without_lmr, original);
    }

    #[test]
    fn null_move_pruning_searches_fewer_nodes() {
        let original = Position::startpos();
        let history = [original.zobrist_hash()];
        let mut with_null_move = original.clone();
        let mut without_null_move = original.clone();

        let pruned = search_with_options(
            &mut with_null_move,
            5,
            &MaterialEvaluator,
            &history,
            SearchOptions::default(),
        );
        let full = search_with_options(
            &mut without_null_move,
            5,
            &MaterialEvaluator,
            &history,
            SearchOptions {
                use_null_move: false,
                ..SearchOptions::default()
            },
        );

        assert!(
            pruned.nodes < full.nodes,
            "null-move pruning should reduce work: {} vs {} nodes",
            pruned.nodes,
            full.nodes
        );
        assert!(pruned.null_move.attempts > 0);
        assert!(pruned.null_move.cutoffs > 0);
        assert!(pruned.null_move.cutoffs <= pruned.null_move.attempts);
        assert_eq!(full.null_move, NullMoveStats::default());
        assert_eq!(with_null_move, original);
        assert_eq!(without_null_move, original);
    }

    #[test]
    fn null_move_pruning_zugzwang_guard_requires_non_pawn_material() {
        let pawn_ending = Position::from_fen("8/8/8/3k4/8/3P4/3K4/8 w - - 0 1").expect("valid FEN");
        let knight_ending =
            Position::from_fen("8/8/8/3k4/8/3N4/3K4/8 w - - 0 1").expect("valid FEN");

        assert!(!has_non_pawn_material(&pawn_ending, Color::White));
        assert!(has_non_pawn_material(&knight_ending, Color::White));
    }

    #[test]
    fn adaptive_null_move_uses_a_larger_reduction_only_at_deep_nodes() {
        assert_eq!(null_move_reduction(6, true), 2);
        assert_eq!(null_move_reduction(7, true), 3);
        assert_eq!(null_move_reduction(20, true), 3);
        assert_eq!(null_move_reduction(20, false), 2);
    }

    #[test]
    fn delta_pruning_skips_a_capture_that_cannot_reach_alpha() {
        let position = Position::from_fen("q3k3/8/8/8/3p4/2P5/8/4K3 w - - 0 1").expect("valid FEN");
        let capture = position
            .generate_legal_moves()
            .into_iter()
            .find(|mv| mv.from() == "c3".parse().unwrap() && mv.to() == "d4".parse().unwrap())
            .expect("c3xd4 should be legal");
        let mut state = SearchState::default();

        assert!(should_delta_prune(
            &position, capture, 0, 401, false, &mut state
        ));
        assert_eq!(state.delta_pruning.attempts, 1);
        assert_eq!(state.delta_pruning.pruned, 1);
    }

    #[test]
    fn delta_pruning_is_disabled_in_low_material_endings() {
        let position = Position::from_fen("4k3/8/8/8/3p4/2P5/8/4K3 w - - 0 1").expect("valid FEN");
        let capture = position
            .generate_legal_moves()
            .into_iter()
            .find(|mv| mv.from() == "c3".parse().unwrap() && mv.to() == "d4".parse().unwrap())
            .expect("c3xd4 should be legal");
        let mut state = SearchState::default();

        assert!(!should_delta_prune(
            &position, capture, 0, 401, false, &mut state
        ));
        assert_eq!(state.delta_pruning, DeltaPruningStats::default());
    }

    #[test]
    fn use_quiescence_false_reintroduces_the_horizon_effect() {
        // Same position as `quiescence_avoids_the_horizon_effect_of_a_
        // losing_trade`, but with `UseQuiescence` off: depth-1 negamax
        // now evaluates the position with material alone the instant it
        // hits depth 0, the same way it would before quiescence existed
        // -- so it should walk right into the rook-for-knight trade that
        // quiescence would otherwise see through.
        let mut position =
            Position::from_fen("4k3/8/4p3/3n4/3R4/8/8/4K3 w - - 0 1").expect("valid FEN");
        let history = [position.zobrist_hash()];

        let result = search_with_options(
            &mut position,
            1,
            &MaterialEvaluator,
            &history,
            SearchOptions {
                use_tt: true,
                use_quiescence: false,
                use_enhanced_quiescence: true,
                use_lmr: true,
                use_null_move: true,
                use_adaptive_null_move: true,
                use_delta_pruning: true,
                use_see: true,
            },
        );

        let best_move = result.best_move.expect("should find a move");
        assert_eq!(
            (best_move.from(), best_move.to()),
            ("d4".parse().unwrap(), "d5".parse().unwrap()),
            "with quiescence off, depth 1 should walk into the losing trade quiescence normally avoids"
        );
    }

    #[test]
    fn enhanced_quiescence_searches_quiet_check_evasions() {
        let mut position =
            Position::from_fen("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1").expect("valid FEN");
        assert!(position.in_check());

        let mut baseline_nodes = 0;
        let mut baseline_path = vec![position.zobrist_hash()];
        let mut baseline_state = SearchState::new(SearchOptions {
            use_enhanced_quiescence: false,
            ..SearchOptions::default()
        });
        quiescence(
            &mut position,
            -SCORE_INF,
            SCORE_INF,
            0,
            0,
            &MaterialEvaluator,
            &mut baseline_nodes,
            &Deadline::none(),
            &mut baseline_path,
            &mut baseline_state,
        )
        .unwrap();

        let mut enhanced_nodes = 0;
        let mut enhanced_path = vec![position.zobrist_hash()];
        let mut enhanced_state = SearchState::default();
        quiescence(
            &mut position,
            -SCORE_INF,
            SCORE_INF,
            0,
            0,
            &MaterialEvaluator,
            &mut enhanced_nodes,
            &Deadline::none(),
            &mut enhanced_path,
            &mut enhanced_state,
        )
        .unwrap();

        assert_eq!(baseline_nodes, 1, "capture-only quiescence stands pat");
        assert!(
            enhanced_nodes > baseline_nodes,
            "enhanced quiescence must search the legal king evasions"
        );
    }
}
