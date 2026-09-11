# mamba_mcts_native

An lc0-style PUCT MCTS for Bee-Mamba, written in Rust and called from
Python via [PyO3](https://pyo3.rs). NN inference deliberately stays in
Python (PyTorch) rather than moving into Rust too -- PyTorch's own backend
selection (CPU / MPS / CUDA) is far more mature than any Rust ML crate we
tried, and this crate's own benchmarking found real inference speed
differences of 2-4x between backends depending on hardware. Only the parts
that benefit from leaving Python -- the tree/PUCT/FPU/virtual-loss logic,
chess move generation, and the NN-evaluation cache -- are native.

See `src/lib.rs`'s module docstring for the full design rationale and the
specific things that were tried and measured (batching, allocation
avoidance, the NN-eval cache, `scan_backend` choice) in the course of
getting this fast. Headline number on an Apple M-series MPS backend: **~1190
simulations/sec** at 800 simulations/move (up from ~230/sec for a naive
single-leaf-at-a-time Python implementation) -- see that docstring for the
full breakdown of what each optimization step bought.

## Building

Requires a Rust toolchain (<https://rustup.rs>) and
[maturin](https://www.maturin.rs/):

```bash
uv pip install maturin
```

This crate's `pyproject.toml` declares `requires-python = ">=3.14"` to
match the rest of this repo, but [PyO3 0.23 doesn't officially support
3.14 yet](https://github.com/PyO3/pyo3/issues) -- building against a 3.14
interpreter needs the stable-ABI forward-compatibility escape hatch:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

(Drop the env var once PyO3 ships official 3.14 support.)

`training/pyproject.toml` depends on this package via a local path source
(see `[tool.uv.sources]` there), so `cd training && uv sync` builds it
automatically as part of the training environment -- the env var above
still applies for now; export it before running `uv sync` if your
interpreter is 3.14.

## Usage

```python
import mamba_mcts_native

# evaluate_batch: Callable[[np.ndarray (n, 64, 20)], tuple[np.ndarray (n, 4096), np.ndarray (n, 128)]]
# history: FENs of every position the real game passed through before `fen`,
# oldest first (not including `fen` itself) -- optional, but without it the
# search can't tell a genuine threefold-repetition/fifty-move draw apart from
# a position its own NN-eval cache has simply seen before, and will happily
# repeat a won position into a draw. See `lib.rs`'s `build_root_counts`.
best_move_uci = mamba_mcts_native.search(
    fen, evaluate_batch, simulations=800, batch_size=32, history=history
)

# ...or with an NN-eval-cache hit/miss breakdown:
best_move_uci, stats = mamba_mcts_native.search_with_stats(
    fen, evaluate_batch, simulations=800, batch_size=32, history=history
)
print(stats.cache_hits, stats.cache_misses)
```

See `training/src/bee_training/chess_mamba/play_mcts.py` for the full UCI
engine built on top of this (checkpoint loading, the `evaluate_batch`
callback, the UCI protocol loop).
