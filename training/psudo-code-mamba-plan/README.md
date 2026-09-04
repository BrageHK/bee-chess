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

Since the SSM ray scan is 20-170x slower than plain attention at board
scale (see benchmark below), `ChessMamba` also supports a **hybrid**
layout: a couple of `SpatialMixer` layers near the input, plain attention
(`AttentionMixer`) for the rest (`model.py`'s `hybrid_layer_types`) — see
"Hybrid architecture" below for the real-data numbers.

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
| `attention_mixer.py` | `AttentionMixer`: plain multi-head self-attention over the 64 squares, drop-in alternative to `SpatialMixer` for the hybrid architecture below. |
| `model.py` | `ChessMamba`: full model — embedding, stacked blocks (each `"ssm"` or `"attn"` via `layer_types`), from-to policy head, HL-Gauss-style value head. `hybrid_layer_types()` builds the recommended mix. |
| `encode.py` | Minimal FEN → `(64, in_dim)` plane encoder + target extraction from this project's self-play `PositionRecord` schema — just enough to run real `data/main-dawg/` positions through the model (see Known limitations: not the validated Phase 4 encoder). |
| `benchmark.py` | Honest speed check against plain attention at board scale (one mixer in isolation), on CPU and GPU. |
| `hybrid_benchmark.py` | Whole-model speed check (pure-SSM vs. hybrid vs. pure-attention `ChessMamba`), on a real training step (forward+backward+Adam) over real `data/main-dawg/` positions. `--light` flag for a quick GPU-only, fewer-iteration run. |

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

## Hybrid architecture: SSM where it might matter, attention everywhere else

None of the tricks above change the fundamental problem: at L=64, plain
attention is close to the cheapest operation there is, and paying the SSM
tax in *every* layer for an unproven inductive-bias bet is expensive —
running `SpatialMixer` in all 8 layers of a small `ChessMamba` costs
~25x what running it in just 1-2 layers does (see table below).

This is not a workaround, it's the standard pattern real hybrid SSM/
attention models use (Jamba, Griffin/Hawk, Zamba): interleave a few SSM
layers among many attention layers (or vice versa), not an all-or-nothing
choice. `model.py`'s `hybrid_layer_types(n_layers, n_ssm)` builds this —
`n_ssm` `SpatialMixer` layers first (closest to the input, where the raw
"stops at first blocker" line-of-sight extraction plausibly matters
most), plain `AttentionMixer` (`attention_mixer.py`) for the rest.

**Real-data training-step benchmark** (`hybrid_benchmark.py --light`):
`d_model=192`, 8 layers, batch 64, on the RX 7900 XTX — a full forward +
backward + Adam step, on **real self-play positions from
`data/main-dawg/`** (via `encode.py`), not random tensors:

```
[cuda] batch=64, d_model=192, n_layers=8 -- real data, real backprop
  pure-ssm (8 ssm)                          388.7 ms/step     165 pos/sec  10.80h per 100k steps  11,066,816 params
  pure-ssm (8 ssm) + compile                255.6 ms/step     250 pos/sec   7.10h per 100k steps  11,066,816 params
  hybrid (2 ssm, 6 attn)                     108.5 ms/step     590 pos/sec   3.01h per 100k steps   4,195,136 params
  hybrid (2 ssm, 6 attn) + compile            73.0 ms/step     876 pos/sec   2.03h per 100k steps   4,195,136 params
  hybrid (1 ssm, 7 attn)                      62.0 ms/step   1,032 pos/sec   1.72h per 100k steps   3,049,856 params
  hybrid (1 ssm, 7 attn) + compile            42.6 ms/step   1,502 pos/sec   1.18h per 100k steps   3,049,856 params
  hybrid (1 ssm, 7 attn, triton)              55.1 ms/step   1,161 pos/sec   1.53h per 100k steps   3,049,856 params
  hybrid (1 ssm, 7 attn, triton) + compile    35.6 ms/step   1,799 pos/sec   0.99h per 100k steps   3,049,856 params
  pure-attn (8 attn)                          15.5 ms/step   4,133 pos/sec   0.43h per 100k steps   1,904,576 params
  pure-attn (8 attn) + compile                 12.4 ms/step   5,147 pos/sec   0.35h per 100k steps   1,904,576 params
```

**The best hybrid config (1 SSM layer + Triton scan + compile) trains at
0.99h per 100k steps — 2.8x slower than pure attention, but 7.2x *faster*
than pure-SSM's 7.1h**, while still keeping the inductive-bias bet alive
in the one layer closest to the input. That's the actual trade this
project is making: pay a real but bounded speed cost to keep testing
whether the geometry-aware scan helps at all (Phase 6/7), instead of
either paying it 8x over or abandoning the hypothesis entirely.

Param counts aren't matched across configs above (pure-ssm's 11M vs.
pure-attn's 1.9M at the same `d_model` — each `SpatialMixer` layer has
many more independent weight matrices: 4 line families × 2 directions ×
a full `MambaBlock` each, vs. one `AttentionMixer`'s single QKV+out
projection) — a real Phase 6 comparison would need to control for that,
same as the plan's own matched-parameter baseline methodology.

## Wiring this up to real training

`encode.py` + `data/main-dawg/` (this project's own self-play/Stockfish
pipeline, `bee_training.dataset`) get real positions through the model
for benchmarking, but real training still needs:

1. **A real board encoder**, not `encode.py`'s minimal one: game-history
   planes (`n_history` past plies, not just the current position) and the
   side-to-move board flip (see Chessformer §3.1 / §A.2) — both skipped
   in `encode.py` for simplicity. `encode.py` also uses a fixed 8-slot aux
   layout rather than the plan's full castling(4)+en-passant(9)+halfmove
   (1)+repetition(1+n_history) breakdown (Section 2) — `ChessMamba`'s
   `in_dim` formula already assumes a fixed "+8", so a real encoder needs
   that formula revisited too, not just the encoder.
2. **More/better data**: `data/main-dawg/` is this project's own
   self-play data; the DeepMind `ChessBench` dataset (Stockfish-annotated
   action-values, `google-deepmind/searchless_chess` on GitHub) is the
   published-comparison choice if you want numbers directly comparable to
   AC-9M/136M/270M.
3. **Real losses**: `encode.py`'s value target is a simple linear cp
   binning, not the HL-Gauss transform over win% Section 6 calls for
   (already scaffolded on the value head, just needs the real target
   transform); the policy target (from-square*64+to-square vs. the
   oracle move) is already what Section 6 describes.

## Known limitations

- Neither scan backend closes the gap to plain attention at this
  sequence length — see benchmark above for the honest cost.
- The `triton_scan` backend is CUDA/ROCm only (no CPU path) and newer/
  less-battle-tested than `mambapy.pscan`, which is why it's opt-in
  (`scan_backend="triton"`) rather than the default.
- `TemporalHistoryMamba.step()` is a stub — real incremental single-ply
  inference needs Mamba's actual recurrent state API, not the batched
  `forward()` used here for clarity.
- `encode.py` is a minimal single-position encoder (no history planes, no
  side-to-move flip, simple linear value binning) built to run real data
  through the model for benchmarking — not the validated Phase 4 encoder
  real training needs (see "Wiring this up to real training").
- `AttentionMixer` is plain attention, no GAB/smolgen-style dynamic
  positional bias — a natural follow-up for the hybrid's attention layers
  if Phase 6/7's own ablation says the extra parameters are worth it.
- Nothing here has been trained on a real objective. All tests
  (`tests/chess_mamba/`) are shape/gradient/masking/locality/encoding
  sanity checks (`test_encode.py` validates against hand-picked FENs),
  not chess-strength evaluation.
