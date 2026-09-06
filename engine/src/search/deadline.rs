//! A cheap, poll-based time budget for search, plus (see `StopSignal`)
//! external cancellation for UCI `stop`/shutdown. Negamax checks a
//! `Deadline` periodically and unwinds early when time is up or a stop
//! has been requested, rather than search needing to be interrupted
//! from outside via a thread primitive of its own.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// How many nodes to visit between `Instant::now()` calls. Checking
/// every single node would make the clock syscall a meaningful
/// fraction of search time; checking too rarely makes the deadline
/// imprecise. This is a small power of two so the modulo check
/// compiles to a cheap bitmask.
const CHECK_INTERVAL_NODES: u64 = 2048;

/// A cheaply cloneable, thread-shared "please stop now" flag -- the
/// mechanism a UCI `stop` (or a shutdown) uses to cancel a search
/// running on another thread (see `crate::uci`'s event loop). Checked
/// on every node (not gated to `CHECK_INTERVAL_NODES` like the wall-
/// clock check -- see `Deadline::is_expired`'s docs): an atomic load is
/// cheap enough to afford unconditionally, and an explicit `stop`
/// should take effect immediately, not wait for the next clock-check
/// boundary.
#[derive(Debug, Clone, Default)]
pub struct StopSignal(Arc<AtomicBool>);

impl StopSignal {
    /// A fresh signal, not yet requested -- one per `go`, so a
    /// previous search's cancellation can never leak into the next
    /// one (see `crate::uci`'s event loop, which makes a new
    /// `StopSignal` for every `go`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests that any search sharing this signal stop as soon as
    /// it next checks (see `Deadline::is_expired`) -- idempotent, and
    /// takes effect for every clone of this signal, not just this one.
    pub fn request_stop(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    #[must_use]
    pub fn is_stop_requested(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A point in time search must stop by, plus an optional external
/// `StopSignal`. `Deadline::none()` means "no time limit and no
/// external cancellation" (used by fixed-depth search, e.g. in tests),
/// which never reports as expired.
#[derive(Debug, Clone)]
pub struct Deadline {
    at: Option<Instant>,
    stop: Option<StopSignal>,
}

impl Deadline {
    /// No time limit and no external cancellation: `is_expired` never
    /// returns `true`.
    #[must_use]
    pub fn none() -> Self {
        Deadline {
            at: None,
            stop: None,
        }
    }

    /// A deadline `budget` from now, with no external cancellation.
    /// See `with_stop_signal` to also attach one.
    #[must_use]
    pub fn from_now(budget: std::time::Duration) -> Self {
        Deadline {
            at: Some(Instant::now() + budget),
            stop: None,
        }
    }

    /// Returns a copy of this deadline that also treats `stop` as an
    /// expiry condition -- see `StopSignal`'s docs. Chainable with
    /// `from_now`/`none` (a `go infinite` search, say, has no `at` but
    /// still needs to honor `stop`).
    #[must_use]
    pub fn with_stop_signal(mut self, stop: StopSignal) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Whether an external stop has been requested (checked
    /// unconditionally -- see `StopSignal`'s docs) or, if `nodes`
    /// visited so far justifies an actual clock check (see
    /// `CHECK_INTERVAL_NODES`), whether the wall-clock deadline has
    /// passed. Cheap to call on every node: the common case (no stop
    /// signal attached or not yet requested, and not yet at a clock-
    /// check boundary) is one branch and one atomic load, no syscall.
    #[must_use]
    pub fn is_expired(&self, nodes: u64) -> bool {
        if let Some(stop) = &self.stop {
            if stop.is_stop_requested() {
                return true;
            }
        }
        let Some(at) = self.at else {
            return false;
        };
        if !nodes.is_multiple_of(CHECK_INTERVAL_NODES) {
            return false;
        }
        Instant::now() >= at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn none_never_expires() {
        let deadline = Deadline::none();
        assert!(!deadline.is_expired(0));
        assert!(!deadline.is_expired(CHECK_INTERVAL_NODES));
        assert!(!deadline.is_expired(u64::MAX));
    }

    #[test]
    fn does_not_check_the_clock_off_the_interval_boundary() {
        // An already-passed deadline still reports "not expired" for
        // node counts that aren't a check boundary, since is_expired
        // is defined to only actually look at the clock then.
        let deadline = Deadline::from_now(Duration::from_secs(0));
        assert!(!deadline.is_expired(1));
        assert!(!deadline.is_expired(CHECK_INTERVAL_NODES - 1));
    }

    #[test]
    fn expires_at_a_check_boundary_once_the_budget_has_passed() {
        let deadline = Deadline::from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));
        assert!(deadline.is_expired(CHECK_INTERVAL_NODES));
    }

    #[test]
    fn does_not_expire_before_the_budget_has_passed() {
        let deadline = Deadline::from_now(Duration::from_secs(60));
        assert!(!deadline.is_expired(CHECK_INTERVAL_NODES));
    }

    #[test]
    fn a_fresh_stop_signal_is_not_requested() {
        let stop = StopSignal::new();
        assert!(!stop.is_stop_requested());
    }

    #[test]
    fn request_stop_is_visible_on_every_clone() {
        let stop = StopSignal::new();
        let clone = stop.clone();

        clone.request_stop();

        assert!(
            stop.is_stop_requested(),
            "stop must be visible through the original handle too"
        );
    }

    #[test]
    fn deadline_with_stop_signal_expires_immediately_on_request_even_with_no_time_limit() {
        // No `at` at all (like `go infinite`'s deadline would have),
        // but a `StopSignal` still makes this expire the moment it's
        // requested -- and, crucially, at any node count, not gated to
        // CHECK_INTERVAL_NODES the way the wall-clock check is.
        let stop = StopSignal::new();
        let deadline = Deadline::none().with_stop_signal(stop.clone());
        assert!(!deadline.is_expired(1), "not requested yet");

        stop.request_stop();

        assert!(
            deadline.is_expired(1),
            "an off-boundary node count must still see the stop"
        );
    }

    #[test]
    fn deadline_with_stop_signal_still_honors_the_wall_clock_deadline() {
        // Attaching a StopSignal that's never requested must not
        // interfere with the ordinary wall-clock expiry path.
        let stop = StopSignal::new();
        let deadline = Deadline::from_now(Duration::from_millis(0)).with_stop_signal(stop);
        std::thread::sleep(Duration::from_millis(5));
        assert!(deadline.is_expired(CHECK_INTERVAL_NODES));
    }
}
