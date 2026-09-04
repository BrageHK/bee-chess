# ChessMamba — a selective-SSM searchless chess architecture

A working prototype that swaps the attention-based geometric mixer
(GAB / smolgen, from the Chessformer / Leela BT-series line of work) for
selective state-space (Mamba/S6) scans, run along the actual lines that
rook, bishop, and queen attacks travel. This is **not** a published
architecture — it's a from-this-conversation hypothesis, implemented and
smoke-tested but not trained or evaluated on real chess data. Treat it as
a starting point for experiments, not a result.

## The idea

A rook/bishop/queen's attack travels in a straight line and stops at the
first piece in the way. A selective SSM scan along that exact line is a
natural fit for that: the input-dependent gate (`dt` in `mamba_core.py`)
can learn to shrink toward zero once it "sees" an occupied square, so
information stops flowing past a blocker — a mechanism dot-product
attention (even with GAB's dynamic bias) never explicitly has.

Knight moves aren't collinear with any straight line, so they get a
separate fixed-adjacency graph mixer instead (`KnightGraphMixer`).

A second, unrelated use of Mamba is included for game history
(`temporal_mixer.py`): instead of Chessformer's fixed n=7-ply concatenated
window, a Mamba scan along the *move* axis keeps a running, unbounded
game-context vector that updates in O(1) per new ply. This is the
direction where Mamba's actual selling point (cheap long sequences)
plausibly matters for chess — unlike the within-position case.

## Files

| File | Contents |
|---|---|
| `geometry.py` | Precomputed rank/file/diagonal/anti-diagonal line indices and the knight-move adjacency table. Run standalone to sanity-check them. |
| `mamba_core.py` | Minimal pure-PyTorch selective SSM (`SelectiveSSM`, `MambaBlock`). Not the official CUDA kernel — see caveats below. |
| `spatial_mixer.py` | `SpatialMixer`: the drop-in replacement for GAB+attention. Runs a shared Mamba scan per line family, both directions, plus the knight graph mixer. |
| `temporal_mixer.py` | `TemporalHistoryMamba`: Mamba over the ply axis for unbounded game history. |
| `model.py` | `ChessMamba`: full model — embedding, stacked blocks, from-to policy head, HL-Gauss-style value head. |
| `benchmark.py` | Honest speed check against plain attention at board scale. |

## Honest benchmark result

At `d_model=128`, batch 8, sequence length 64 (one board), on CPU:

```
SpatialMixer (SSM, ours):   760.85 ms / fwd+bwd   (1,101,952 params)
Plain multi-head attention:   4.91 ms / fwd+bwd      (66,048 params)
SSM is ~155x the wall-clock time of plain attention at this scale.
```

This is expected, not a bug: attention over 64 tokens is already about as
cheap as an operation gets, and this implementation runs a Python-level
sequential scan over many small (≤8-step) line-batches, which has real
per-call overhead that a fused kernel wouldn't. **The pitch for this
architecture is inductive bias (explicit line-of-sight), not speed.** If
you want to pursue this seriously:

- swap `mamba_core.MambaBlock` for the official `mamba_ssm.Mamba` CUDA
  block (drop-in shape-compatible) and re-benchmark on GPU — it will
  close some of this gap, though a 4-8 step scan is still short enough
  that it may never beat attention on raw speed;
- or accept the speed cost and evaluate purely on whether the inductive
  bias improves puzzle accuracy / sample efficiency versus GAB at equal
  parameter count, which is the actual open question here, not throughput.

## Wiring this up to real training

This repo only has the model — you'd still need to:

1. **Board encoder**: convert a `python-chess` `Board` (or FEN) into the
   `(64, in_dim)` plane tensor `ChessMamba.forward` expects — one-hot
   piece-per-square for the current position and last `n_history` plies,
   plus castling/en-passant/rule-50 scalars broadcast across squares
   (see Chessformer §3.1 / §A.2 for the exact recipe this follows).
2. **Data**: the DeepMind `ChessBench` dataset (Stockfish-annotated
   action-values, `google-deepmind/searchless_chess` on GitHub) is the
   natural choice, since it's public and lets you compare directly
   against the published AC-9M/136M/270M numbers.
3. **Losses**: bin the Stockfish win% into `n_value_bins` classes and
   train the value head with HL-Gauss or plain cross-entropy (already
   scaffolded); train the policy head with cross-entropy over the
   flattened `(64*64)` from-to logits against the oracle's UCI move.

## Known limitations of this prototype

- Pure-Python sequential scan — fine for correctness, bad for throughput
  (see benchmark above).
- `TemporalHistoryMamba.step()` is a stub — real incremental single-ply
  inference needs Mamba's actual recurrent state API, not the batched
  `forward()` used here for clarity.
- Nothing here has been trained. All tests are shape/gradient/masking
  sanity checks, not chess-strength evaluation.
