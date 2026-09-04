# ChessMamba — a selective-SSM searchless chess architecture

**The implementation now lives at `src/bee_training/chess_mamba/`**, with
real pytest coverage under `tests/chess_mamba/` — this file and
`CHESSMAMBA_PLAN.md` stay here as the architecture spec/design doc. This
is **not** a published architecture — it's a from-this-conversation
hypothesis, implemented and tested but not trained or evaluated on real
chess data. Treat it as a starting point for experiments, not a result.

The scan backend is `mambapy.pscan` (pure PyTorch, Blelloch parallel
scan) rather than a hand-rolled Python loop or the official `mamba-ssm`
CUDA kernel — the latter does not build on RDNA3/gfx1100 ROCm (RX 7900
class cards), only on datacenter Instinct cards, so `mambapy` is the
implementation that's actually runnable here. See
`src/bee_training/chess_mamba/mamba_core.py`.

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

All under `src/bee_training/chess_mamba/`:

| File | Contents |
|---|---|
| `geometry.py` | Precomputed rank/file/diagonal/anti-diagonal line indices and the knight-move adjacency table. Run standalone to sanity-check them. |
| `mamba_core.py` | Selective SSM (`SelectiveSSM`, `MambaBlock`), scan computed via `mambapy.pscan` (parallel scan, pure PyTorch — no custom CUDA/HIP kernel, so it runs on ROCm unmodified). |
| `spatial_mixer.py` | `SpatialMixer`: the drop-in replacement for GAB+attention. Runs a shared Mamba scan per line family, both directions, plus the knight graph mixer. |
| `temporal_mixer.py` | `TemporalHistoryMamba`: Mamba over the ply axis for unbounded game history. |
| `model.py` | `ChessMamba`: full model — embedding, stacked blocks, from-to policy head, HL-Gauss-style value head. |
| `benchmark.py` | Honest speed check against plain attention at board scale, on CPU and GPU. |

## Honest benchmark result

At `d_model=128`, batch 8, sequence length 64 (one board), measured on an
AMD RX 7900 XTX (ROCm) and its host CPU:

```
[cpu]  SpatialMixer (SSM, ours):  84.32 ms / fwd+bwd  (1,101,952 params)
[cpu]  Plain multi-head attn:      0.87 ms / fwd+bwd     (66,048 params)
[cpu]  SSM is 96.8x the wall-clock time of plain attention at L=64.

[cuda] SpatialMixer (SSM, ours):  11.11 ms / fwd+bwd  (1,101,952 params)
[cuda] Plain multi-head attn:      0.31 ms / fwd+bwd     (66,048 params)
[cuda] SSM is 36.4x the wall-clock time of plain attention at L=64.
```

Still slower than plain attention, as expected: attention over 64 tokens
is already about as cheap as an operation gets, and each ray scan is only
4-8 steps — too short to amortize any implementation's per-call overhead.
The parallel-scan backend (`mambapy.pscan`, replacing a naive Python-loop
scan) roughly halves the gap versus a sequential loop, and the GPU
closes it further (36x vs 97x on CPU), but doesn't erase it. **The pitch
for this architecture stays inductive bias (explicit line-of-sight), not
speed.**

The official `mamba_ssm.Mamba` CUDA kernel (what you'd reach for on
Nvidia) is not a realistic option here: it requires custom CUDA/HIP
kernels that do not build on RDNA3/gfx1100 consumer cards (RX 7900-class)
as of 2026 — confirmed via multiple open upstream build failures — only
on datacenter Instinct cards (MI200/MI300). `mambapy`'s pure-PyTorch
parallel scan is the pragmatic ROCm-safe choice instead, at some
throughput cost versus a fused kernel.

The actual open question is whether the inductive bias improves puzzle
accuracy / sample efficiency versus GAB at equal parameter count — see
`CHESSMAMBA_PLAN.md` Phase 6/7 for the controlled comparison that would
answer that.

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

## Known limitations

- `mambapy.pscan` is a parallel scan but still not a fused kernel — see
  benchmark above for the honest cost of that versus plain attention.
- `TemporalHistoryMamba.step()` is a stub — real incremental single-ply
  inference needs Mamba's actual recurrent state API, not the batched
  `forward()` used here for clarity.
- Nothing here has been trained. All tests (`tests/chess_mamba/`) are
  shape/gradient/masking/locality sanity checks, not chess-strength
  evaluation.
