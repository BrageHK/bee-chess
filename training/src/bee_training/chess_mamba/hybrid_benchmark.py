"""Whole-model speed check: pure-SSM vs. hybrid (a couple of SpatialMixer
layers, rest AttentionMixer) vs. pure-attention `ChessMamba`, at a
realistic config scale (see CHESSMAMBA_PLAN.md Section 10's "small"
config). Unlike `benchmark.py` (which isolates one mixer), this measures
a full forward+backward training step through the whole stacked model,
including both heads and the combined policy+value loss -- on real
self-play positions from `data/main-dawg/` (via `encode.py`), not random
tensors, so the loss is real and the gradients are real, not just
shape-compatible noise.

Run as: `python -m bee_training.chess_mamba.hybrid_benchmark` (full: CPU+GPU)
or `... --light` (GPU only, fewer timed iterations -- still real data, real
backward passes, real Adam steps, and still compiled; CPU pure-ssm at this
scale is minutes-slow with no interesting story of its own, so --light
just skips straight to the GPU numbers that matter, faster).
"""

import glob
import sys
import time

import torch
import torch.nn.functional as F

from bee_training.chess_mamba.encode import IN_DIM, load_real_batch
from bee_training.chess_mamba.model import N_SQUARES, ChessMamba, hybrid_layer_types
from bee_training.chess_mamba.triton_scan import TRITON_AVAILABLE

D_MODEL = 192
N_LAYERS = 8
BATCH = 64
N_VALUE_BINS = 128
N_ITERS = 15
DATA_GLOB = "data/main-dawg/shards/*.positions.jsonl"


def _make_step(model, x, target_move, target_bin, optimizer=None):
    """A realistic training step, not just forward+backward: includes the
    optimizer update too, since that's what real training time is spent
    on -- the loss.backward()-only number understates a training loop by
    however long Adam's own update takes."""

    def step():
        policy_logits, value_logits = model(x)
        loss = F.cross_entropy(policy_logits.reshape(x.shape[0], -1), target_move) \
            + F.cross_entropy(value_logits, target_bin)
        loss.backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad()

    return step


def bench(model, x, target_move, target_bin, optimizer=None, n_iters=N_ITERS):
    step = _make_step(model, x, target_move, target_bin, optimizer=optimizer)
    for _ in range(3):
        step()
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        step()
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000


def bench_compiled(model, x, target_move, target_bin, optimizer=None, n_iters=N_ITERS):
    """Returns None on a torch.compile/Inductor failure instead of raising --
    compile is best-effort (see README: it already failed to compile the
    pure-ssm config once with an Inductor assertion on this exact setup),
    and one config's compiler bug shouldn't take down the whole comparison."""
    try:
        compiled = torch.compile(model)
        step = _make_step(compiled, x, target_move, target_bin, optimizer=optimizer)
        t0 = time.perf_counter()
        step()
        if x.is_cuda:
            torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        steady_ms = bench(compiled, x, target_move, target_bin, optimizer=optimizer, n_iters=n_iters)
        return compile_s, steady_ms
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        return None, f"compile failed: {type(e).__name__}: {e}".splitlines()[0][:80]


def _print_row(name, ms_or_msg, n_params, compile_s=None):
    if isinstance(ms_or_msg, str):
        print(f"  {name:34s} {ms_or_msg}")
        return
    ms = ms_or_msg
    steps_per_sec = 1000 / ms
    positions_per_sec = BATCH * steps_per_sec
    hours_per_100k_steps = 100_000 / steps_per_sec / 3600
    warm = f"  (compile warmup {compile_s:5.1f}s)" if compile_s is not None else ""
    print(f"  {name:34s} {ms:9.3f} ms/step  {positions_per_sec:9,.0f} pos/sec  "
          f"{hours_per_100k_steps:6.2f}h per 100k steps  {n_params:>11,} params{warm}")


def run(device, n_iters=N_ITERS, do_compile=True):
    """Benchmarks a realistic training step (forward + backward + Adam
    update -- not just fwd+bwd) on real self-play positions from
    data/main-dawg/, since that's what a real training loop actually
    spends its time on and actually sees."""
    torch.manual_seed(0)
    shard_paths = sorted(glob.glob(DATA_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"no shards found at {DATA_GLOB} -- run the dataset generator first")
    x, target_move, target_bin = load_real_batch(
        shard_paths, batch_size=BATCH, n_value_bins=N_VALUE_BINS, device=device
    )
    assert x.shape == (BATCH, N_SQUARES, IN_DIM)

    configs = [
        ("pure-ssm (8 ssm)", ["ssm"] * N_LAYERS, "pscan"),
        ("hybrid (2 ssm, 6 attn)", hybrid_layer_types(N_LAYERS, n_ssm=2), "pscan"),
        ("hybrid (1 ssm, 7 attn)", hybrid_layer_types(N_LAYERS, n_ssm=1), "pscan"),
        ("pure-attn (8 attn)", ["attn"] * N_LAYERS, "pscan"),
    ]
    if device == "cuda" and TRITON_AVAILABLE:
        configs.append(("hybrid (1 ssm, 7 attn, triton)", hybrid_layer_types(N_LAYERS, n_ssm=1), "triton"))

    print(f"\n[{device}] batch={BATCH}, d_model={D_MODEL}, n_layers={N_LAYERS}, n_iters={n_iters}, "
          f"{len(shard_paths)} real shard(s) -- realistic training step (fwd+bwd+Adam) on real data")
    for name, layer_types, scan_backend in configs:
        model = ChessMamba(d_model=D_MODEL, n_layers=N_LAYERS, d_state=8, expand=1.0,
                            scan_backend=scan_backend, layer_types=layer_types,
                            n_history=0, n_value_bins=N_VALUE_BINS).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        ms = bench(model, x, target_move, target_bin, optimizer=opt, n_iters=n_iters)
        _print_row(name, ms, n_params)

        if device == "cuda" and do_compile:
            model_c = ChessMamba(d_model=D_MODEL, n_layers=N_LAYERS, d_state=8, expand=1.0,
                                  scan_backend=scan_backend, layer_types=layer_types,
                                  n_history=0, n_value_bins=N_VALUE_BINS).to(device)
            opt_c = torch.optim.Adam(model_c.parameters(), lr=1e-4)
            compile_s, ms_c = bench_compiled(model_c, x, target_move, target_bin,
                                              optimizer=opt_c, n_iters=n_iters)
            _print_row(name + " + compile", ms_c, n_params, compile_s=compile_s)
    print()


if __name__ == "__main__":
    if "--light" in sys.argv:
        # GPU only (CPU pure-ssm at this scale is minutes-slow, no
        # interesting story of its own), fewer timed iterations -- still
        # real data, real backward, real Adam, still compiled.
        run("cuda", n_iters=5, do_compile=True)
    else:
        run("cpu", n_iters=N_ITERS, do_compile=True)
        if torch.cuda.is_available():
            run("cuda", n_iters=N_ITERS, do_compile=True)
