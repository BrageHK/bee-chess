# 0001. v1 Engine Architecture

## Status

Accepted

## Context

Bee Chess spans three languages (Rust, Python, TypeScript/React) and several
plausible engine designs (alpha-beta with a classical or NNUE-style
evaluator, MCTS/PUCT with a batched policy/value network, ensembles of
external engines). Without an explicit decision, contributors will each
assume a different target architecture, and interfaces will be built to be
"flexible" instead of correct for one design.

## Decision

Bee Chess v1 is a standard-chess UCI engine.

The competition engine is implemented in Rust.

Python owns training and dataset generation.

React is a development/visualization client and is not part of the
competition hot path.

The initial search architecture is alpha-beta/PVS with an incrementally
updatable neural evaluator.

ONNX may be used as a model interchange/reference runtime, but the engine
architecture must not depend directly on ONNX.

UCI exists only at the process boundary. Internal engine APIs use typed
Rust structures.

HTTP/WebSocket observability lives outside the search hot path.

## Consequences

- Chess960, MCTS/PUCT, batched GPU inference, and ensemble strategies are
  explicitly out of scope for v1 and are tracked as experimental/future
  work, not folded into the primary interfaces.
- Every internal crate boundary (UCI adapter, search controller, search
  algorithm, evaluator) is designed around alpha-beta + incremental
  evaluation first. A future MCTS backend would sit behind the same
  `Search`/`Evaluator` contracts rather than reshaping them.
- No raw UCI strings may appear below the UCI adapter module.
- The engine binary must not require an HTTP server to run or compete.
- This decision can be revisited via a new ADR if a future milestone
  requires it (e.g. an explicit MCTS research effort), but it is binding
  for all v1 work.
- The architecture is not tied to one specific machine: it must remain
  configurable across reasonable CPU, memory, and time-control
  environments rather than assuming a particular thread count, hash
  size, or GPU availability. Hardware-specific tuning and
  competition-specific constraints (baseline opponent, time controls,
  deployment platform) are documented separately, in experiment,
  benchmark, and tournament configuration rather than in this ADR.
