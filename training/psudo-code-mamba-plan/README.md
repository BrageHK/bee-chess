# ChessMamba — a selective-SSM searchless chess architecture

**The implementation now lives at `src/bee_training/chess_mamba/`**, with
real pytest coverage under `tests/chess_mamba/` — this file and
`CHESSMAMBA_PLAN.md` stay here as the architecture spec/design doc. This
is **not** a published architecture — it's a from-this-conversation
hypothesis, implemented and tested but not trained or evaluated on real
chess data. Treat it as a starting point for experiments, not a result.

The scan backend is pluggable (`scan_backend="pscan"` or `"triton"`,
threaded down from `ChessMamba`) rather than a hand-rolled Python loop or
the official `mamba-ssm` CUDA kernel — the latter does not build on
RDNA3/gfx1100 ROCm (RX 7900 class cards), only on datacenter Instinct
cards. `"pscan"` (`mambapy.pscan`, pure PyTorch) is the default — it needs
no custom kernel, so it works on any device. `"triton"`
(`triton_scan.triton_pscan`) is a from-scratch fused Triton kernel, opt-in
because it's CUDA/ROCm-only and newer/less-battle-tested, but faster (see
benchmark below) and confirmed working on this exact GPU. See
`src/bee_training/chess_mamba/mamba_core.py` and `triton_scan.py`.

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
| `mamba_core.py` | Selective SSM (`SelectiveSSM`, `MambaBlock`), scan backend pluggable via `scan_backend` (`"pscan"` default, `"triton"` opt-in — see `SCAN_BACKENDS`/`get_scan_fn`). |
| `triton_scan.py` | `triton_pscan`: fused Triton kernel alternative to `mambapy.pscan`, streams over L instead of materializing `(B,L,D,N)`. CUDA/ROCm only. |
| `spatial_mixer.py` | `SpatialMixer`: the drop-in replacement for GAB+attention. Runs a shared Mamba scan per line family, both directions, plus the knight graph mixer. |
| `temporal_mixer.py` | `TemporalHistoryMamba`: Mamba over the ply axis for unbounded game history. |
| `model.py` | `ChessMamba`: full model — embedding, stacked blocks, from-to policy head, HL-Gauss-style value head. |
| `benchmark.py` | Honest speed check against plain attention at board scale, on CPU and GPU. |

## Honest benchmark result

At `d_model=128`, batch 64, sequence length 64 (one board), measured on an
AMD RX 7900 XTX (ROCm) and its host CPU — `benchmark.py` reconstructs the
original pre-`pscan` Python-loop scan alongside the current defaults, so
these are same-run, same-hardware comparisons across this project's own
history, not numbers from different runs stitched together:

```
[cpu] batch=64, d_model=128, L=64
  loop (pre-pscan, d_state=16 expand=2x, unfused)  1970.871 ms/fwd+bwd    388.3x attention
  pscan (tuned: d_state=8 expand=1x, fused)         329.308 ms/fwd+bwd     64.9x attention
  pscan (tuned + torch.compile)                     446.705 ms/fwd+bwd     88.0x attention  (compile warmup:   4.6s)
  attention (eager)                                   5.076 ms/fwd+bwd      1.0x attention
  attention (torch.compile)                           4.048 ms/fwd+bwd      0.8x attention  (compile warmup:   0.3s)

[cuda] batch=64, d_model=128, L=64
  loop (pre-pscan, d_state=16 expand=2x, unfused)   116.830 ms/fwd+bwd    154.1x attention
  pscan (tuned: d_state=8 expand=1x, fused)          30.578 ms/fwd+bwd     40.3x attention
  pscan (tuned + torch.compile)                      20.391 ms/fwd+bwd     26.9x attention  (compile warmup:   3.7s)
  triton (tuned, fused)                              26.242 ms/fwd+bwd     34.6x attention
  triton (tuned + torch.compile)                     16.159 ms/fwd+bwd     21.3x attention  (compile warmup:  12.1s)
  attention (eager)                                   0.758 ms/fwd+bwd      1.0x attention
  attention (torch.compile)                           0.873 ms/fwd+bwd      1.2x attention  (compile warmup:   0.1s)
```

Reading this honestly:

- **Still slower than plain attention**, as expected — attention over 64
  tokens is already about as cheap as an operation gets, and each ray
  scan is only 4-8 steps, too short to amortize any implementation's
  per-call overhead. **The pitch for this architecture stays inductive
  bias (explicit line-of-sight), not speed.**
- **The gap is memory-bandwidth-bound, not launch-overhead-bound**: cost
  scales close to linearly with `d_state` and the expand factor (halving
  either roughly halves wall-clock time), because `pscan` materializes
  full `(B, L, D_inner, N)` tensors in HBM for every scan call. This is
  also why the SSM/attention ratio got *worse*, not better, at bigger
  batch sizes in earlier testing (34x → 133x → 117x at B=8/64/256,
  eventually OOM-ing at B=1024 on this 24GB card) — more batch means more
  of that same large tensor, not better amortization of a fixed cost.
- **Four real, stacked wins over the original loop-based prototype**, all
  shape/gradient/mask-tested (`tests/chess_mamba/`), none changing the
  architecture: (1) `mambapy.pscan` (parallel scan, O(log L) instead of a
  Python `for` loop) in place of a hand-rolled sequential scan; (2)
  `d_state` 16→8 and SSM expand 2x→1x — cost scales ~linearly with both,
  and there's no more evidence an 8-step board-line scan needs that
  capacity than there was for the 4x FFN expansion LC0's own ablations
  found unnecessary; (3) fusing each line family's forward+backward pass
  into one scan call instead of two (same math — the scan has no
  cross-batch interaction, so concatenating along the batch dim and
  splitting after is exactly equivalent, just fewer scan invocations);
  (4) a fused Triton kernel (`triton_scan.py`, opt-in via
  `scan_backend="triton"`) that streams over L instead of materializing
  `(B,L,D,N)` — the same trick the official CUDA kernel uses, from
  scratch, confirmed working on gfx1100. Combined (Triton + compile):
  **~7.2x faster than the original prototype on GPU** (116.8ms → 16.2ms).
- **Triton alone is a modest ~1.2x over tuned `pscan`** (30.6ms → 26.2ms),
  well short of the 2-4.8x seen when benchmarking the scan in isolation —
  once wrapped in the real `in_proj`/`x_proj`/`dt_proj`/`out_proj` linears
  and the gather/scatter indexing, those now dominate more of the total
  cost than the scan itself. Squeezing further would mean going after the
  linears/indexing next (e.g. fusing `in_proj` across all 8 scan calls
  into one batched matmul), not another scan-kernel rewrite.
- **`torch.compile`'s payoff depends on backend and workload — benchmark
  it, don't assume it**: it helps both scan backends on GPU (pscan 1.5x,
  triton 1.6x) but is 1.3x *slower* on CPU for our model (329ms → 447ms),
  while for plain attention it's close to a wash either way. Compiling
  the attention baseline too (not just our own model) is what makes this
  an honest comparison.

The official `mamba_ssm.Mamba` CUDA kernel (what you'd reach for on
Nvidia) is not a realistic option here: it requires custom CUDA/HIP
kernels that do not build on RDNA3/gfx1100 consumer cards (RX 7900-class)
as of 2026 — confirmed via multiple open upstream build failures — only
on datacenter Instinct cards (MI200/MI300). The Triton kernel above is
the from-scratch equivalent — Triton's AMD backend does work on this
GPU, at some remaining gap versus what a hand-tuned fused CUDA kernel on
Nvidia hardware would achieve.

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

- Neither scan backend closes the gap to plain attention at this
  sequence length — see benchmark above for the honest cost.
- The `triton_scan` backend is CUDA/ROCm only (no CPU path) and newer/
  less-battle-tested than `mambapy.pscan`, which is why it's opt-in
  (`scan_backend="triton"`) rather than the default.
- `TemporalHistoryMamba.step()` is a stub — real incremental single-ply
  inference needs Mamba's actual recurrent state API, not the batched
  `forward()` used here for clarity.
- Nothing here has been trained. All tests (`tests/chess_mamba/`) are
  shape/gradient/masking/locality sanity checks, not chess-strength
  evaluation.
