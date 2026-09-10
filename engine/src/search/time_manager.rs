//! Turns a UCI `go` command's clock fields into concrete search
//! deadlines, so nothing below this module ever needs to know what
//! `wtime`/`btime`/`winc`/`binc`/`movestogo` mean.
//!
//! Bee does not maintain its own chess clock: every `go` carries the
//! authoritative current time for both sides (per the UCI protocol),
//! and [`ClockTimeControl`] captures just the side-relative slice of
//! that a single move decision needs -- "my time left, my increment,
//! how many moves (if known) I still have to make it through."
//! [`allocate_time`] turns that into a [`TimeBudget`]: a *soft* target
//! (iterative deepening should not start another depth once this has
//! passed) and a *hard* limit (search must abort mid-depth rather than
//! cross this, no matter what). See each type's own docs for why the
//! split matters -- a single deadline can't express both "normally
//! stop around here" and "never, ever cross this" at once.
//!
//! [`allocate_time`] is a pure function of its inputs -- no clocks, no
//! sleeping -- specifically so the allocation math can be unit tested
//! directly, without timing-sensitive integration tests.

use std::time::Duration;

/// The side-relative clock state a single `go` command carries, once
/// UCI's `wtime`/`btime`/`winc`/`binc`/`movestogo` fields have already
/// been resolved to "my side's" numbers -- nothing downstream of this
/// needs to know which color is on move. `None` throughout means no
/// clock was given at all (e.g. `go depth 8`, `go infinite`): time
/// management has nothing to allocate, and callers should not invoke
/// [`allocate_time`] in that case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockTimeControl {
    /// Time left on the mover's own clock right now, per the most
    /// recent `go`'s `wtime`/`btime`.
    pub time_left: Duration,
    /// Increment the mover's clock gains after making this move, per
    /// `winc`/`binc`. Zero if the time control has no increment.
    pub increment: Duration,
    /// Moves remaining until the next time control, per `movestogo`,
    /// if the GUI supplied it. `None` means an unknown/effectively
    /// unbounded horizon (most common for increment-only or "whole
    /// game" time controls), in which case [`allocate_time`] falls
    /// back to `TimeManagerConfig::estimated_moves_remaining`.
    pub moves_to_go: Option<u32>,
}

/// Tunable constants for [`allocate_time`], kept separate from
/// [`ClockTimeControl`] since these describe *policy* (how cautious to
/// be) rather than anything the GUI told us about the actual clock.
/// Exposed as a UCI option only for `move_overhead` for now (see
/// `crate::engine`'s `MoveOverhead` option) -- the rest are constants
/// until real measurement suggests they should be tunable too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeManagerConfig {
    /// Fixed slice of every move's time budget reserved for protocol/
    /// process/network overhead -- the time between "search decides on
    /// a move" and "the GUI/server actually sees `bestmove`". Bee must
    /// never plan to use this time for thinking: on a real server
    /// (e.g. Lichess) it's the difference between "flagged" and
    /// "didn't." Configurable via the `MoveOverhead` UCI option, since
    /// the right value depends on the deployment (a local GUI needs
    /// far less than a network round trip to a lichess-bot bridge).
    pub move_overhead: Duration,
    /// Always-untouchable slice of `time_left` itself, on top of
    /// `move_overhead`, so a string of moves that each slightly
    /// underestimate their own overhead can never collectively run the
    /// clock to zero. Unlike `move_overhead` (spent every move),
    /// `emergency_reserve` is compared against the *remaining* clock
    /// and only ever constrains allocation as time gets low.
    pub emergency_reserve: Duration,
    /// How many moves to assume remain in the game when `movestogo`
    /// isn't given. Deliberately conservative (see the module docs --
    /// getting the exact right number is future, measurement-driven
    /// work; having *a* reasonable floor is what matters for v1).
    pub estimated_moves_remaining: u32,
    /// The hard limit is this many times the soft target, clamped so
    /// it can never eat into `emergency_reserve`. A book-hit or
    /// otherwise trivial move finishing well under the soft target is
    /// normal and fine; this only bounds how far a *slow* iteration is
    /// allowed to run past the soft target before being aborted.
    pub hard_limit_multiplier: u32,
}

impl Default for TimeManagerConfig {
    fn default() -> Self {
        Self {
            move_overhead: Duration::from_millis(DEFAULT_MOVE_OVERHEAD_MS),
            emergency_reserve: Duration::from_millis(50),
            estimated_moves_remaining: 30,
            hard_limit_multiplier: 3,
        }
    }
}

/// How much headroom `estimate_next_depth_cost`-based prediction
/// requires beyond the estimated cost itself before allowing a new
/// depth to start -- i.e. a depth is only started if `estimated_cost *
/// PREDICTION_SAFETY_MARGIN` still fits in the remaining hard budget.
/// Deliberately conservative (reserving a further ~15% beyond the
/// estimate itself) since the estimate is a rough one (see
/// `estimate_next_depth_cost`'s docs on its own clamping) and the
/// actual cost of getting this wrong -- crossing the hard deadline
/// anyway -- already has its own backstop (the hard deadline check
/// mid-iteration), but avoiding that backstop actually having to fire
/// is the entire point of predicting in the first place.
const PREDICTION_SAFETY_MARGIN: f64 = 1.15;

/// How much more expensive the *next* depth is assumed to be, relative
/// to the last completed one, when there's no real growth-factor
/// history yet (only one depth has completed so far -- see
/// `estimate_next_depth_cost`'s docs). A conservative middle estimate:
/// low enough not to refuse a second depth almost every move, high
/// enough to reflect that iterative deepening's cost realistically
/// does grow, not shrink or stay flat, per ply.
const DEFAULT_GROWTH: f64 = 2.0;

/// Clamp bounds for the growth factor `estimate_next_depth_cost`
/// computes from two real completed depths' timings -- a single
/// unusually quiet or unusually tactical transition between two
/// specific depths shouldn't be allowed to dominate the estimate
/// either direction: `MIN_GROWTH` keeps a rare depth-over-depth
/// *shrink* (possible with move-ordering/TT effects) from making the
/// next depth look free, and `MAX_GROWTH` keeps a rare explosive
/// transition from making every subsequent depth this move look
/// permanently unaffordable.
const MIN_GROWTH: f64 = 1.5;
const MAX_GROWTH: f64 = 4.0;

/// Estimates how long the *next* depth will take, in milliseconds,
/// from `last_depth_ms` (the most recently completed depth's own
/// wall-clock time) and `previous_depth_ms` (the depth before that,
/// `None` if only one depth has completed so far this search). Pure
/// arithmetic -- no clock reads -- so, like `allocate_time`, this is
/// directly unit-testable without timing-sensitive tests.
///
/// The growth factor between the last two completed depths (clamped to
/// `[MIN_GROWTH, MAX_GROWTH]`) is projected forward one more ply;
/// `DEFAULT_GROWTH` stands in for that ratio when there's no second
/// data point yet. This is deliberately the simplest model that uses
/// real information already on hand (see the module docs on why a
/// boring, measurable first version beats a more elaborate unmeasured
/// one) -- not a branching-factor model of chess search specifically,
/// just "depths have been getting more expensive at roughly this rate,
/// assume that continues."
#[must_use]
pub fn estimate_next_depth_cost(last_depth_ms: u64, previous_depth_ms: Option<u64>) -> u64 {
    let growth = match previous_depth_ms {
        Some(previous) if previous > 0 => {
            (last_depth_ms as f64 / previous as f64).clamp(MIN_GROWTH, MAX_GROWTH)
        }
        _ => DEFAULT_GROWTH,
    };
    (last_depth_ms as f64 * growth).round() as u64
}

/// Whether a depth estimated to cost `estimated_next_depth_ms` should
/// be started at all, given `elapsed_ms` already spent this search and
/// `hard_ms` the absolute ceiling -- see `PREDICTION_SAFETY_MARGIN`'s
/// docs for why the estimate is padded before comparing. Pure
/// arithmetic, like `estimate_next_depth_cost` itself.
#[must_use]
pub fn next_depth_is_affordable(
    elapsed_ms: u64,
    estimated_next_depth_ms: u64,
    hard_ms: u64,
) -> bool {
    let padded_estimate = estimated_next_depth_ms as f64 * PREDICTION_SAFETY_MARGIN;
    elapsed_ms as f64 + padded_estimate <= hard_ms as f64
}

/// Nodes visited per second, from a total node count and the wall-
/// clock time it took -- the single number both the live `nps` UCI
/// field and `TimeManagementTelemetry::avg_nps` are built from, kept
/// as one pure, unit-tested function so the two call sites (a running
/// search reporting its current rate, and a finished search reporting
/// its overall average) can never compute it inconsistently. Returns
/// `0` for a zero (or, defensively, negative-rounding) elapsed time
/// rather than dividing by zero -- a search that completes in under a
/// millisecond has no meaningfully measurable rate.
#[must_use]
pub fn nodes_per_second(nodes: u64, elapsed: Duration) -> u64 {
    let elapsed_ms = elapsed.as_millis();
    if elapsed_ms == 0 {
        return 0;
    }
    (u128::from(nodes) * 1000 / elapsed_ms) as u64
}

/// Default `move_overhead`, in milliseconds -- shared with the `uci`
/// module's advertised `MoveOverhead` UCI option default, so the two
/// can never drift apart.
pub const DEFAULT_MOVE_OVERHEAD_MS: u64 = 30;

/// The two deadlines iterative deepening actually needs -- see
/// `search::search_iterative_with_options`'s docs for how each is
/// used. `soft <= hard` always holds.
///
/// * `soft`: once this has elapsed, don't start another iteration --
///   play whatever the last *completed* depth found. Iterative
///   deepening's cost roughly doubles per ply, so stopping at (rather
///   than exactly on) the soft target is normal and expected: the
///   soft target bounds when a *new* iteration begins, not when the
///   current one must finish.
/// * `hard`: the absolute ceiling. If an iteration is still running
///   when this is reached, it must be aborted (its result discarded,
///   per `search_to_depth`'s existing contract) rather than allowed to
///   run any longer -- this is the deadline that must never be
///   crossed, since crossing it risks losing on time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeBudget {
    pub soft: Duration,
    pub hard: Duration,
}

/// Version of the `bee-tm` telemetry line's field set -- bump this
/// whenever a field is removed or changes meaning (adding a new field
/// is not a breaking change: consumers are required to ignore unknown
/// keys, per this type's docs). Emitted as `v=1` so a consumer parsing
/// old logs, or built against a future engine version, can tell which
/// fields to expect rather than guessing from what happens to be
/// present.
pub const BEE_TM_VERSION: u32 = 1;

/// Per-move time-management telemetry: everything about *how* a clock-
/// bounded search actually spent its budget that only the engine
/// itself can know (Lab separately measures what it, not the engine,
/// is authoritative for -- real wall-clock elapsed, clock remaining,
/// increment applied, timeout/result; see `lab::game`). This is
/// deliberately not folded into `SearchResult`/the ordinary `info
/// depth ...` line: those describe *what search found*, this describes
/// *how the time budget was spent finding it* -- e.g. `aborted`
/// carries a wall-clock duration for a depth whose search result was
/// itself discarded and never became part of any `SearchResult` at
/// all.
///
/// Rendered as a single `info string bee-tm ...` line (see
/// `to_bee_tm_line`) -- a machine-readable record, not human-oriented
/// diagnostic prose (contrast `crate::diagnostics`, which is prose by
/// design). Consumers must treat this as append-only: recognize the
/// `bee-tm` prefix, split on whitespace then each token once on `=`,
/// and silently ignore any key they don't recognize -- see
/// `BEE_TM_VERSION`'s docs on why a missing/unknown field is expected,
/// forward-compatible behavior, not an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimeManagementTelemetry {
    /// This move's allocated `TimeBudget`, unpacked -- what
    /// `allocate_time` decided *before* any searching happened.
    pub soft_ms: u64,
    pub hard_ms: u64,
    /// The depth iterative deepening actually completed and reported
    /// as its `SearchResult` -- same number `info depth` itself
    /// carries, duplicated here purely so a `bee-tm` line is a
    /// complete record on its own without needing to be correlated
    /// with a separate `info depth` line to be useful.
    pub completed_depth: u32,
    /// Wall-clock time spent on the one iteration (if any) that was
    /// started but never completed -- cut off by the hard deadline (or
    /// an external `stop`) partway through, its result discarded per
    /// `search_to_depth`'s contract. Zero if every started iteration
    /// completed (the common case: the soft deadline, not the hard
    /// one, is what normally ends a search). This is the single most
    /// actionable number for deciding whether depth-cost prediction is
    /// worth building: time here is pure waste, spent computing a
    /// result that was then thrown away.
    pub aborted_ms: u64,
    /// How many times the best move at the root changed between one
    /// completed depth and the next (i.e. `depth N`'s best move
    /// differs from `depth N-1`'s) -- a stability signal: a search
    /// that keeps agreeing with itself as it goes deeper is a good
    /// candidate for stopping early; one that keeps flip-flopping
    /// probably isn't.
    pub best_move_changes: u32,
    /// Signed centipawn difference between the last two completed
    /// depths' scores (`last - previous`, from the root side to
    /// move's own perspective both times) -- `None` if fewer than two
    /// depths completed. A second stability signal alongside
    /// `best_move_changes`: a small score delta at the end of a search
    /// suggests the evaluation has settled; a large swing on the final
    /// depth suggests it might not have.
    pub score_delta_cp: Option<i32>,
    /// Overall nodes-per-second rate for the whole search: total nodes
    /// visited across every *completed* depth (an aborted, discarded
    /// iteration contributes no nodes here, same as it contributes no
    /// nodes to any `SearchResult`) divided by wall-clock time since
    /// the search began (`search_start.elapsed()`, not any single
    /// depth's own duration) -- see `nodes_per_second`. This is the
    /// measurement `estimate_next_depth_cost`'s pure wall-time growth
    /// model deliberately doesn't need (see that function's docs), but
    /// it's exactly what a human or tool watching `bee-tm` lines needs
    /// to sanity-check the engine's actual throughput on a given
    /// position/machine.
    pub avg_nps: u64,
}

impl TimeManagementTelemetry {
    /// Renders this record as the payload of an `info string bee-tm
    /// ...` line (the `info string bee-tm ` prefix itself is the
    /// caller's job -- see `crate::uci`'s docs on where diagnostics vs.
    /// structured `info` fields are written) -- see this type's docs
    /// on the wire format's stability rules.
    #[must_use]
    pub fn to_bee_tm_line(self) -> String {
        let mut line = format!(
            "v={BEE_TM_VERSION} soft_ms={} hard_ms={} completed_depth={} aborted_ms={} best_move_changes={} avg_nps={}",
            self.soft_ms,
            self.hard_ms,
            self.completed_depth,
            self.aborted_ms,
            self.best_move_changes,
            self.avg_nps,
        );
        if let Some(delta) = self.score_delta_cp {
            line.push_str(&format!(" score_delta_cp={delta}"));
        }
        line
    }
}

/// Computes a [`TimeBudget`] for one move from `control` and `config`.
/// Pure and deterministic -- no clock reads, no sleeping -- so this is
/// exactly as unit-testable as any other arithmetic (see this module's
/// tests).
///
/// The algorithm is deliberately boring (see the module docs): spend
/// roughly `usable_time / moves_remaining`, plus a fraction of the
/// increment (since that time is replenished every move regardless of
/// how this one goes), as the soft target; allow up to
/// `hard_limit_multiplier` times that as the hard limit, clamped so it
/// never eats into `emergency_reserve`. Getting the constants exactly
/// right is future, measurement-driven work (see the module docs on
/// stability-based time management); having the soft/hard split and
/// the reserve/overhead accounting correct is what matters here.
#[must_use]
pub fn allocate_time(control: ClockTimeControl, config: &TimeManagerConfig) -> TimeBudget {
    // Never plan to spend the reserve or the per-move overhead --
    // `usable` is the only time this function will ever allocate from.
    let protected = config.move_overhead + config.emergency_reserve;
    let usable = control.time_left.saturating_sub(protected);

    let moves_remaining = control
        .moves_to_go
        .filter(|&n| n > 0)
        .unwrap_or(config.estimated_moves_remaining)
        .max(1);

    let base = usable / moves_remaining;
    // Half the increment: the other half is slack for the *next*
    // move's own allocation, rather than this move greedily spending
    // all of it up front.
    let increment_bonus = control.increment / 2;
    let soft = (base + increment_bonus).min(usable);

    let hard = soft
        .saturating_mul(config.hard_limit_multiplier)
        .min(usable);

    TimeBudget { soft, hard }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(time_left_ms: u64, increment_ms: u64, moves_to_go: Option<u32>) -> ClockTimeControl {
        ClockTimeControl {
            time_left: Duration::from_millis(time_left_ms),
            increment: Duration::from_millis(increment_ms),
            moves_to_go,
        }
    }

    #[test]
    fn bee_tm_line_includes_the_version_and_every_field() {
        let telemetry = TimeManagementTelemetry {
            soft_ms: 220,
            hard_ms: 660,
            completed_depth: 7,
            aborted_ms: 201,
            best_move_changes: 3,
            score_delta_cp: Some(-42),
            avg_nps: 1_500_000,
        };

        assert_eq!(
            telemetry.to_bee_tm_line(),
            "v=1 soft_ms=220 hard_ms=660 completed_depth=7 aborted_ms=201 best_move_changes=3 avg_nps=1500000 score_delta_cp=-42"
        );
    }

    #[test]
    fn bee_tm_line_omits_score_delta_when_fewer_than_two_depths_completed() {
        let telemetry = TimeManagementTelemetry {
            soft_ms: 220,
            hard_ms: 660,
            completed_depth: 1,
            aborted_ms: 0,
            best_move_changes: 0,
            score_delta_cp: None,
            avg_nps: 0,
        };

        let line = telemetry.to_bee_tm_line();
        assert!(!line.contains("score_delta_cp"), "got: {line}");
        // Every other field is still present -- a missing key is only
        // ever the *consumer's* signal to skip it, not a reason for
        // the emitter to omit anything it does have.
        assert!(line.contains("completed_depth=1"));
    }

    #[test]
    fn bee_tm_line_has_no_whitespace_inside_any_value() {
        // Consumers split the whole line on whitespace first, then
        // each token once on '=' -- see the type's docs on the wire
        // format's stability rules. A value containing a space would
        // silently corrupt that parse.
        let telemetry = TimeManagementTelemetry {
            soft_ms: 220,
            hard_ms: 660,
            completed_depth: 7,
            aborted_ms: 0,
            best_move_changes: 1,
            score_delta_cp: Some(12),
            avg_nps: 850_000,
        };

        for token in telemetry.to_bee_tm_line().split(' ') {
            assert_eq!(token.matches('=').count(), 1, "malformed token: {token}");
        }
    }

    #[test]
    fn soft_is_never_greater_than_hard() {
        for (time_left, increment, moves_to_go) in [
            (60_000, 0, None),
            (60_000, 2_000, None),
            (3_000, 0, None),
            (50, 0, None),
            (10_000, 100, Some(1)),
            (10_000, 100, Some(40)),
        ] {
            let budget = allocate_time(
                control(time_left, increment, moves_to_go),
                &TimeManagerConfig::default(),
            );
            assert!(
                budget.soft <= budget.hard,
                "soft ({:?}) must never exceed hard ({:?}) for time_left={time_left}ms increment={increment}ms movestogo={moves_to_go:?}",
                budget.soft,
                budget.hard,
            );
        }
    }

    #[test]
    fn more_increment_gives_a_larger_soft_budget() {
        let config = TimeManagerConfig::default();
        let no_increment = allocate_time(control(60_000, 0, None), &config);
        let with_increment = allocate_time(control(60_000, 2_000, None), &config);

        assert!(with_increment.soft > no_increment.soft);
    }

    #[test]
    fn low_time_left_gives_a_conservative_budget() {
        let config = TimeManagerConfig::default();
        let plenty = allocate_time(control(60_000, 0, None), &config);
        let low = allocate_time(control(3_000, 0, None), &config);

        assert!(low.soft < plenty.soft);
        assert!(low.hard < plenty.hard);
    }

    #[test]
    fn near_zero_time_left_still_produces_a_tiny_but_valid_budget() {
        // 50ms remaining, all of it eaten by move_overhead +
        // emergency_reserve in the default config (30ms + 50ms) --
        // usable time bottoms out at zero, and the budget must reflect
        // that rather than underflowing or panicking.
        let config = TimeManagerConfig::default();
        let budget = allocate_time(control(50, 0, None), &config);

        assert_eq!(budget.soft, Duration::ZERO);
        assert_eq!(budget.hard, Duration::ZERO);
    }

    #[test]
    fn movestogo_one_allows_spending_much_more_than_the_default_horizon() {
        let config = TimeManagerConfig::default();
        let default_horizon = allocate_time(control(60_000, 0, None), &config);
        let last_move_before_control = allocate_time(control(60_000, 0, Some(1)), &config);

        assert!(last_move_before_control.soft > default_horizon.soft);
    }

    #[test]
    fn hard_limit_never_exceeds_usable_time() {
        // A large hard_limit_multiplier must still be clamped by
        // usable time -- the hard limit can never eat into the
        // move_overhead/emergency_reserve that's supposed to be
        // permanently protected.
        let config = TimeManagerConfig {
            hard_limit_multiplier: 100,
            ..TimeManagerConfig::default()
        };
        let budget = allocate_time(control(1_000, 0, None), &config);
        let protected = config.move_overhead + config.emergency_reserve;

        assert!(budget.hard <= Duration::from_millis(1_000).saturating_sub(protected));
    }

    #[test]
    fn zero_moves_to_go_is_treated_like_unknown_rather_than_dividing_by_zero() {
        // A malformed/defensive `movestogo 0` must not panic or
        // allocate an unbounded budget -- fall back to the configured
        // estimate exactly as if movestogo had been omitted.
        let config = TimeManagerConfig::default();
        let unknown = allocate_time(control(60_000, 0, None), &config);
        let zero = allocate_time(control(60_000, 0, Some(0)), &config);

        assert_eq!(unknown, zero);
    }

    #[test]
    fn estimate_next_depth_cost_uses_default_growth_with_only_one_data_point() {
        // No `previous_depth_ms` yet (only depth 1 has completed) --
        // falls back to `DEFAULT_GROWTH` (2.0).
        assert_eq!(estimate_next_depth_cost(100, None), 200);
    }

    #[test]
    fn estimate_next_depth_cost_projects_the_observed_growth_ratio() {
        // Depth grew from 50ms to 100ms (2x) -- project the same 2x
        // ratio forward.
        assert_eq!(estimate_next_depth_cost(100, Some(50)), 200);
    }

    #[test]
    fn estimate_next_depth_cost_clamps_an_extreme_growth_ratio() {
        // A 20x jump between two real depths must not be projected
        // forward as-is -- clamped to MAX_GROWTH (4.0).
        assert_eq!(estimate_next_depth_cost(2000, Some(100)), 8000);
    }

    #[test]
    fn estimate_next_depth_cost_clamps_a_shrinking_ratio_to_the_minimum() {
        // Depth 5 taking *less* time than depth 4 (plausible with
        // move-ordering/TT effects) must not make the next depth look
        // free -- clamped to MIN_GROWTH (1.5), not treated as < 1.
        assert_eq!(estimate_next_depth_cost(50, Some(100)), 75);
    }

    #[test]
    fn estimate_next_depth_cost_treats_a_zero_previous_depth_as_no_history() {
        // A previous depth reported as 0ms (plausible for a trivial
        // position/depth) can't produce a meaningful ratio -- falls
        // back to DEFAULT_GROWTH rather than dividing by zero.
        assert_eq!(estimate_next_depth_cost(100, Some(0)), 200);
    }

    #[test]
    fn next_depth_is_affordable_when_the_padded_estimate_fits() {
        // 50ms elapsed, estimate 100ms, padded to 115ms (1.15x) --
        // fits comfortably in a 300ms hard budget.
        assert!(next_depth_is_affordable(50, 100, 300));
    }

    #[test]
    fn next_depth_is_affordable_is_false_when_the_padded_estimate_does_not_fit() {
        // 200ms elapsed, estimate 100ms (padded to 115ms) -- 200 + 115
        // = 315ms, past a 300ms hard budget.
        assert!(!next_depth_is_affordable(200, 100, 300));
    }

    #[test]
    fn nodes_per_second_divides_nodes_by_elapsed_seconds() {
        assert_eq!(nodes_per_second(2_000_000, Duration::from_secs(2)), 1_000_000);
    }

    #[test]
    fn nodes_per_second_is_zero_for_zero_elapsed_time() {
        // A search finishing in under a millisecond has no
        // meaningfully measurable rate -- must not divide by zero.
        assert_eq!(nodes_per_second(1_000, Duration::ZERO), 0);
    }

    #[test]
    fn nodes_per_second_handles_sub_second_elapsed_time() {
        assert_eq!(nodes_per_second(500_000, Duration::from_millis(500)), 1_000_000);
    }

    #[test]
    fn next_depth_is_affordable_accounts_for_the_safety_margin_at_the_boundary() {
        // Without the 1.15x safety margin, 200 + 100 = 300 would
        // exactly fit -- the margin must make this reject it.
        assert!(!next_depth_is_affordable(200, 100, 300));
    }
}
