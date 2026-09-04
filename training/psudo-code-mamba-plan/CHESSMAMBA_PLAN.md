# ChessMamba — architecture spec & implementation plan

**Status:** unproven hypothesis, not a published architecture. Motivated by a
gap in the current literature (Chessformer/GAB, Leela's smolgen, the original
Grandmaster-Level Chess Without Search paper) rather than by any existing
result. Treat every design choice below as "our best guess," and treat
Section 12 (first experiment) as the actual point of building this — the
plan is designed to get you a real yes/no signal as cheaply as possible,
not to jump straight to a big training run.

---

## 1. Core design thesis

Rook, bishop, and queen attacks travel in a straight line and stop at the
first piece in the way. Chessformer's GAB (and Leela's smolgen before it)
give the model a *dynamic positional bias* conditioned on the board state,
but that bias is generated once per forward pass from a compressed summary
— it doesn't have an explicit mechanism for "propagate along this line
until you hit something." A selective state-space model's input-dependent
gate (Mamba's `Δ`) is a natural fit for exactly that: it can learn to
collapse toward zero the instant it "sees" an occupied square, cutting off
propagation past a blocker.

Knight moves aren't collinear with any straight line, so they get a
separate, fixed-adjacency mechanism instead of a scan.

Two independent uses of Mamba are in scope:

- **(A) Spatial** — within one position, scanning along the four line
  families a sliding piece can attack on. This is the main architectural
  bet and the focus of Phases 1–3 and 6–8.
- **(B) Temporal** — across plies, compressing arbitrary-length game
  history into a running state that updates in O(1) per move, replacing
  Chessformer's fixed 7-ply concatenation window. This is where Mamba's
  actual selling point (cheap long sequences) plausibly matters — a single
  64-square position is already cheap for plain attention, so (A) is a
  bet on inductive bias, not speed, while (B) is a bet on both. Scoped as
  a stretch goal (Phase 9) so it doesn't block getting a result on (A).

---

## 2. Input representation

Square-token board encoding — 64 tokens, one per square — **not** the
original paper's FEN-character tokenization. Square tokens are what make a
geometry-aware mixer possible at all.

Per-square input vector, concatenated then linearly embedded to `d_model`:

```
one-hot(12 piece types) × (1 + n_history) planes      -> 12*(1+n_history) dims
castling rights (4 booleans)                          -> 4 dims
en-passant target file (one-hot over 8 files + none)  -> 9 dims
halfmove clock / 100                                  -> 1 dim
repetition flag per included ply (1 + n_history)       -> (1+n_history) dims
-----------------------------------------------------------------
in_dim = 12*(1+n_history) + 4 + 9 + 1 + (1+n_history)
```

Flip the board to the perspective of the side to move (standard trick —
means the model only ever has to learn "my pieces vs. their pieces," not
color-specific behavior).

`n_history = 7` by default — Chessformer found a large accuracy jump from
0→7 plies and no significant further gain at 31, so 7 is the cheapest
setting that captures the benefit.

---

## 3. Core primitive: selective SSM (S6) block

Standard Mamba recurrence, per-channel diagonal state:

```
A = -exp(A_log)                     # (D_inner, D_state), learned, stable (negative)
Δ = softplus(dt_proj(x_proj(x)))    # (B, L, D_inner), input-dependent — the "selection"
B_t, C_t = split(x_proj(x))         # (B, L, D_state) each, also input-dependent
Ābar_t = exp(Δ_t * A)                # (B, L, D_inner, D_state)
B̄bar_t = Δ_t * B_t                   # (B, L, D_inner, D_state)  [Euler discretization]
h_t = Ābar_t * h_{t-1} + B̄bar_t * x_t
y_t = C_t · h_t + D_skip * x_t
```

**Masking for padded/variable-length lines:** force `Δ = 0` at padded steps
(multiply by the mask before the softplus's output is used). That makes
`Ābar = 1` (state passes through unchanged) and `B̄bar = 0` (no new input
mixed in) — a padded step is then an exact no-op, not an approximate one.
This is the mechanism that lets rank/file lines (always length 8) and
diagonals (length 1–8) share the same scan code. **Verify this with a unit
test that deliberately puts garbage values in the padded region and checks
the real-region output is bit-for-bit unaffected** — cheap to write, and
it's the kind of masking bug that fails silently otherwise.

Two implementation options:

1. **Pure PyTorch, Python-level `for` loop over the scan.** Fine for
   correctness — every sequence we scan here is ≤64 steps (board lines are
   ≤8, game history in Phase 9 might run to a few hundred plies at most).
   No need for a parallel-scan algorithm at this length. Use this for
   Phases 1–6 so the whole stack stays CPU-testable without a CUDA
   dependency.
2. **Official `mamba-ssm` package** (`pip install mamba-ssm`, needs
   `causal-conv1d`, needs a CUDA GPU). Swap in for Phase 8 once you're
   ready to actually chase training-time throughput. Interface-compatible
   as long as you keep the same `(B, L, D)` in/out contract.

Interfaces:

```
SelectiveSSM(d_inner, d_state=16).forward(x: (B,L,D_inner), mask: (B,L) optional) -> (B,L,D_inner)
MambaBlock(d_model, d_inner=2*d_model, d_state=16).forward(x: (B,L,D_model), mask=None) -> (B,L,D_model)
    # in_proj -> split (x, z) -> SelectiveSSM(x) -> * silu(z) -> out_proj
```

---

## 4. Board geometry (precomputed, not learned)

Square index convention: `sq = rank*8 + file`, both 0–7 (a1=0 … h8=63).

Four line families, each built once at model-construction time as
`(idx, mask)` buffer pairs:

| Family | # lines | Line length | Padding needed? |
|---|---|---|---|
| rank | 8 | 8 | no |
| file | 8 | 8 | no |
| main diagonal (`rank-file` = const) | 15 | 1–8 | yes, pad to 8 |
| anti-diagonal (`rank+file` = const) | 15 | 1–8 | yes, pad to 8 |

Sanity-check values to hardcode as regression tests once built:

- Diagonal line lengths, in offset order, should read
  `[8,7,6,5,4,3,2,1,7,6,5,4,3,2,1]` for one family and the mirror
  `[1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]` for the other — this is just the
  well-known "how long is each diagonal of a chessboard" pattern, so any
  deviation means an indexing bug.
- Order each diagonal's squares by increasing rank so "forward" and
  "backward" scan direction are well-defined and consistent across all
  15 lines in a family.

**Knight adjacency:** a `(64, 8)` padded index table + mask, built from the
8 knight-move offsets, keeping only in-bounds destinations per square.
Regression-test value: **average out-degree must equal exactly 5.25** —
this follows from the fact that an empty 8×8 board has exactly 336 total
knight moves (336/64 = 5.25), so it's a strong, easy correctness check.

---

## 5. Spatial mixer (the main architectural bet)

For each of the four line families, run **one shared `MambaBlock`
instance** (not one per individual line — share weights across all 8 rank
lines, etc.) applied twice: once forward, once on the sequence-and-mask
reversed (then flip the output back). Sum the two directions.

Data flow per family:

```
board_feats: (B, 64, D)
  -> gather via idx into (B, num_lines, max_len, D)      [advanced indexing]
  -> zero out padded slots via mask
  -> reshape to (B*num_lines, max_len, D)
  -> MambaBlock_fwd(flat, mask=tiled_mask)                -> y_fwd
  -> MambaBlock_bwd(flat.flip(dim=1), mask=mask.flip(-1)).flip(dim=1) -> y_bwd
  -> y = (y_fwd + y_bwd) * mask
  -> scatter back to (B, 64, D) via index_add_ on flattened idx
```

**Knight graph mixer:** small multi-head attention restricted to each
square's ≤8 knight-neighbors (gather keys/values via the adjacency table,
masked softmax over the valid neighbors only, `-inf`-mask the rest before
softmax). This is the one piece of the mixer that stays attention-based,
since knight relations aren't a scan problem.

**Merge:** concatenate the four ray-family outputs and the knight-mixer
output (5 × `d_model`) and project back down to `d_model` with one linear
layer.

```
SpatialMixer(d_model, d_state=16).forward(board_feats: (B,64,D)) -> (B,64,D)
```

---

## 6. ChessMamba block & full model

```
ChessMambaBlock:
  x = x + SpatialMixer(LayerNorm(x))
  x = x + FFN(LayerNorm(x))            # expansion ratio 1x by default —
                                        # LC0's own ablations found ~no
                                        # benefit from the usual 4x for
                                        # chess transformers; only raise
                                        # this if your own ablation shows
                                        # it matters
```

Full model:

```
embed:       Linear(in_dim -> d_model)
body:        N x ChessMambaBlock
final_norm:  LayerNorm(d_model)
```

**Value head:** mean-pool over the 64 squares → LayerNorm → 2-layer MLP →
`n_value_bins` logits (128 by default, matching the original paper). Train
with HL-Gauss / categorical cross-entropy over binned Stockfish win%, not
MSE regression — this is already a validated win, not part of the
hypothesis under test, so don't re-litigate it.

**Policy head:** attention-style "from-square, to-square" head
(Chessformer's design, architecture-agnostic) — project the encoder output
to query ("from") and key ("to") vectors, scaled dot product gives a
`(64, 64)` move-logit matrix, flatten to a 4096-way classification target
against the oracle's UCI move. Mask illegal moves before softmax at
eval/inference time (not necessarily needed in the training loss itself,
same as the original paper's approach).

---

## 7. Temporal history variant (stretch goal — Phase 9)

A second, separate `MambaBlock` scanning per-ply pooled position summaries
along the *move* axis, forward/causal only (a game only flows one
direction, unlike a board line). Produces a running game-context vector
that updates in O(1) per new ply, instead of Chessformer's fixed n=7-ply
concatenation window.

Flag clearly to whoever implements this: batched `forward()` replay over
the whole history (as in a first pass) is fine for *training*, but a real
live-play engine needs Mamba's genuine recurrent single-step API (carry
`(h, conv_state)` explicitly between moves) to get the O(1)-per-move
property in practice — don't let "it works in training" be mistaken for
"it's fast at inference" without that follow-up piece.

---

## 8. Suggested repo layout

```
chess_mamba/
├── geometry.py          # Phase 1 — line families + knight adjacency, pure precompute
├── mamba_core.py         # Phase 1 — SelectiveSSM, MambaBlock
├── spatial_mixer.py       # Phase 2 — ray scans + knight mixer + merge
├── model.py               # Phase 3 — ChessMambaBlock, ChessMamba, both heads
├── temporal_mixer.py       # Phase 9 — TemporalHistoryMamba (stretch)
├── data/
│   ├── encode.py          # Phase 4 — python-chess Board/FEN -> (64, in_dim) planes
│   └── chessbench.py       # Phase 4 — ChessBench loader (or a subset thereof)
├── baselines/
│   └── plain_attention.py  # Phase 6 — matched-param attention baseline for A/B test
├── train.py                # Phase 5 — training loop, losses, checkpointing
├── eval.py                  # Phase 7 — puzzle accuracy, Kendall's τ, action accuracy
├── benchmark.py              # Phase 8 — throughput vs. baseline, CPU and GPU
└── tests/
    ├── test_geometry.py
    ├── test_mamba_core.py
    ├── test_spatial_mixer.py
    └── test_model.py
```

---

## 9. Phased implementation plan

Work through these in order. Each phase has a hard stop — get the
acceptance criteria green before moving on, since a bug in Phase 1 will
silently corrupt everything built on top of it.

### Phase 0 — environment
- `torch`, `pytest`. Optionally `python-chess` (Phase 4), `mamba-ssm` +
  `causal-conv1d` (Phase 8 only, needs a CUDA GPU — skip for now).
- **Done when:** `import torch; import pytest` works.

### Phase 1 — geometry + core SSM primitive
- Implement `geometry.py` and `mamba_core.py` per Sections 3–4.
- **Acceptance criteria:**
  - Diagonal line lengths match the `[8,7,...,1,7,...,1]` pattern exactly.
  - Knight adjacency average out-degree == 5.25 exactly.
  - Padding no-op test: garbage in the masked region of a `MambaBlock`
    input must not change the real-region output at all.
  - Gradients flow through every parameter on a dummy loss.

### Phase 2 — spatial mixer
- Implement `spatial_mixer.py` per Section 5.
- **Acceptance criteria:**
  - `SpatialMixer(d).forward(x: (B,64,d)) -> (B,64,d)`, shape-exact.
  - Gradients flow through every parameter (rays + knight mixer + merge).
  - Swapping the piece on a single square changes that square's own
    output and the outputs of squares on its rank/file/diagonals/knight
    graph, but not unrelated squares (a locality sanity check — write
    this as an actual test, it'll catch scatter-index bugs Phase 1's
    tests can't).

### Phase 3 — full model assembly
- Implement `model.py` per Section 6.
- **Acceptance criteria:**
  - Forward pass on a random dummy batch produces `(B,64,64)` policy
    logits and `(B,n_value_bins)` value logits.
  - A combined policy+value cross-entropy loss backward pass touches
    100% of parameters.
  - Print total parameter count for a couple of configs (see Section 10)
    so you know what you're comparing against before training anything.

### Phase 4 — data pipeline
- `data/encode.py`: convert a `python-chess` `Board`/FEN into the `(64,
  in_dim)` tensor from Section 2. This is the part most likely to have a
  subtle bug (wrong castling-bit order, wrong en-passant encoding, etc.)
  — write a test that round-trips a handful of known positions (starting
  position, a position with all castling rights lost, a position with an
  en-passant target) and checks the encoded planes by hand.
- `data/chessbench.py`: loader for the DeepMind `ChessBench` dataset
  (`google-deepmind/searchless_chess` on GitHub — 10M games, Stockfish
  16 action-value annotations, 15.3B action-value pairs at full size).
  Start with a small slice (10k–100k games) for fast iteration; the full
  dataset is only needed once you're past the ablation stage.
- **Acceptance criteria:** encode → decode round-trip matches on a
  held-out set of hand-picked FENs; loader produces batches of the shape
  Phase 3's model expects with zero reshaping in the training loop.

### Phase 5 — training loop
- Standard supervised setup: Adam, HL-Gauss loss on the value head,
  cross-entropy on the flattened policy logits vs. the oracle move.
  Mirror the original paper's protocol (10M steps at large scale, but
  start far smaller — see Section 12) so results are comparable.
- **Acceptance criteria:** loss decreases on a small (10k-game) slice
  within a few thousand steps; no NaNs; checkpointing works.

### Phase 6 — matched-parameter baseline
- Implement `baselines/plain_attention.py`: the same embedding, same
  heads, same block count, but with a standard multi-head self-attention
  layer (optionally + GAB if you want the strongest possible baseline)
  in place of `SpatialMixer`, parameter-matched as closely as practical.
  This is the whole point of the exercise — the current literature has
  never run this exact controlled comparison, so it's the actual novel
  contribution here, not the architecture in isolation.
- **Acceptance criteria:** baseline and ChessMamba have parameter counts
  within ~5% of each other at each config in Section 10.

### Phase 7 — evaluation harness
- Puzzle accuracy (exact solution-sequence match), action accuracy
  (top-1 vs. oracle), Kendall's τ (rank correlation of predicted vs.
  oracle action ranking) — same three metrics and same puzzle set
  methodology as the original paper's Table 2, so numbers are directly
  comparable to published AC-9M/136M/270M results.
- **Acceptance criteria:** harness runs end to end on both ChessMamba and
  the Phase 6 baseline, producing a single comparison table.

### Phase 8 — efficiency pass (optional, only if Phase 6/7 shows promise)
- Swap `mamba_core.MambaBlock` for the official `mamba-ssm` CUDA block;
  re-run `benchmark.py` on GPU.
- **Acceptance criteria:** documented ms/forward-backward at matched
  batch size and `d_model`, compared honestly against the baseline —
  don't be surprised if it's still slower than plain attention at L=64;
  see Section 11.

### Phase 9 — temporal history (stretch)
- Implement `temporal_mixer.py` per Section 7, wire into the encoder as
  an alternative to the fixed-window history planes, evaluate whether it
  changes human-move-prediction accuracy at low compute cost (this is
  the more obviously-motivated of the two Mamba uses, worth doing if
  Phase 6/7 didn't pan out for the spatial bet but you still want to use
  Mamba somewhere real).

### Phase 10 — interpretability check (stretch)
- If Phase 6/7 shows a real gain: visualize the learned `Δ` (selection
  gate) values along a rank/file/diagonal in positions with a clear
  blocking piece, and check whether it actually collapses toward zero
  past the blocker as the design story predicts. This is the difference
  between "it works" and "it works for the reason we think it does" —
  worth doing before writing anything up.

---

## 10. Starter configs

Start at the smallest scale — comparable to the original paper's 9M model
— so you can iterate in minutes, not hours, before committing to anything
bigger.

| Config | `d_model` | `n_layers` | `d_state` | `n_history` | `n_value_bins` | approx. params |
|---|---|---|---|---|---|---|
| tiny (Phase 1–3 smoke tests) | 64 | 2 | 8 | 7 | 128 | ~0.6M |
| small (Phase 5–7 first ablation) | ~180–200 | 8 | 16 | 7 | 128 | ~9M |
| medium (only if small shows signal) | ~350 | 12 | 16 | 7 | 128 | ~40–50M |

---

## 11. Known risks — carry these forward honestly

- **No evidence yet that this beats GAB.** The mechanistic story (gate =
  learnable occlusion) is plausible, not proven. Phase 6/7 is the actual
  test; don't scale up before it's green.
- **Expect it to be slower than plain attention at board scale, possibly
  even with the CUDA kernel.** A hand-rolled Python-loop pure-PyTorch
  version was benchmarked at ~155x slower than plain multi-head attention
  at `d_model=128`, `L=64` on CPU — expected, since attention over 64
  tokens is already about as cheap as an operation gets, and each ray
  scan is only 4–8 steps long (too short to amortize per-call overhead,
  fused kernel or not). This is a bet on inductive bias / sample
  efficiency, not throughput — frame Phase 7's results that way.
- **The controlled comparison is the actual contribution.** Chessformer's
  own playing-strength track trained on Leela self-play data, not the
  original paper's Stockfish-annotated ChessBench set — nobody has yet
  retrained a GAB-style architecture directly on ChessBench for a true
  apples-to-apples comparison. Phase 6 closes that gap either way,
  regardless of which architecture wins.
- **Knight/king safety concepts may need more than the graph mixer
  provides.** The knight mixer only covers knight-move adjacency; broader
  king-safety reasoning (multiple pieces converging on a square) may
  still need real attention or a second, larger-radius graph — watch for
  this specifically during Phase 10 if results are mixed.

---

## 12. First experiment — what "done" looks like

Smallest test that would actually tell you something: train the **small**
config (Section 10, ~9M params) of both ChessMamba and the Phase 6
plain-attention baseline on the same 1M-game ChessBench slice, same
Stockfish time-limit oracle, same number of steps, then compare puzzle
accuracy, action accuracy, and Kendall's τ on the same held-out puzzle
set — exactly the original paper's own Table 2 ablation methodology. That
comparison, at this small scale, should take a small fraction of the
compute of a full run and is the one number that decides whether Phase
8's efficiency work and Phase 9's stretch goal are worth doing at all.

---

## Suggested Claude Code kickoff

Paste this file into the repo (e.g. as `CHESSMAMBA_PLAN.md`), then start
with something like:

> Implement Phase 0 and Phase 1 from `CHESSMAMBA_PLAN.md`. Write the unit
> tests described in the acceptance criteria before or alongside the
> implementation. Run them, show me the results, and stop — don't start
> Phase 2 until I've looked at Phase 1's output.

Keep each phase its own checkpoint rather than letting the agent run
straight through to Phase 5 — Phase 1's geometry tables are load-bearing
for everything after them, and a silent indexing bug there is much
cheaper to catch now than after a training run.
