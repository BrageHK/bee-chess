//! Evaluator contract.
//!
//! An `Evaluator` scores a position from the side-to-move's perspective.
//! Per ADR 0001, the v1 evaluator is an incrementally updatable neural
//! evaluator; this module only establishes the trait boundary that
//! `search` depends on. Concrete evaluators (classical, NNUE, ONNX
//! reference backend) are implemented in follow-up PRs behind this trait.

use crate::chess::Position;
use crate::search::Score;

/// Scores a position. Implementations must not perform network I/O or
/// other unbounded-latency work on this hot path (see CONTRIBUTING.md).
pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> Score;
}
