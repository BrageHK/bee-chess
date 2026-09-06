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
}
