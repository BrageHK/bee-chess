//! A cheap, poll-based time budget for search, since there's no
//! threading/cancellation infrastructure yet (that's #7's territory).
//! Negamax checks a `Deadline` periodically and unwinds early when
//! time is up, rather than being interrupted from outside.

use std::time::Instant;

/// How many nodes to visit between `Instant::now()` calls. Checking
/// every single node would make the clock syscall a meaningful
/// fraction of search time; checking too rarely makes the deadline
/// imprecise. This is a small power of two so the modulo check
/// compiles to a cheap bitmask.
const CHECK_INTERVAL_NODES: u64 = 2048;

/// A point in time search must stop by. `Deadline::none()` means "no
/// time limit" (used by fixed-depth search, e.g. in tests), which
/// never reports as expired.
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    at: Option<Instant>,
}

impl Deadline {
    /// No time limit: `is_expired` never returns `true`.
    #[must_use]
    pub fn none() -> Self {
        Deadline { at: None }
    }

    /// A deadline `budget` from now.
    #[must_use]
    pub fn from_now(budget: std::time::Duration) -> Self {
        Deadline {
            at: Some(Instant::now() + budget),
        }
    }

    /// Whether `nodes` visited so far justifies an actual clock check
    /// (see `CHECK_INTERVAL_NODES`) and, if so, whether the deadline
    /// has passed. Cheap to call on every node: the common case (not
    /// yet at a check boundary) is a single integer comparison with no
    /// syscall.
    #[must_use]
    pub fn is_expired(&self, nodes: u64) -> bool {
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
}
