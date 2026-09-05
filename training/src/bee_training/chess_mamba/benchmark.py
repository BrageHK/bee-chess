"""Honest speed check: at L=64 (one board), is the SSM mixer actually
faster than plain attention? (Spoiler: it shouldn't be -- the pitch for
this architecture is inductive bias, not speed, at this sequence length.)

Compares several points on this project's own history, all measured in the
same run on the same hardware (CPU and, if available, GPU -- ROCm shows up
as `cuda`):

  - `loop`            : the original prototype's scan -- a Python `for`
                         loop over L, d_state=16, expand=2x, unfused
                         forward/backward passes. Reconstructed here
                         (matches what shipped before `mambapy.pscan` was
                         adopted) purely as a same-run comparison point --
                         not used by the real model.
  - `pscan`            : current production SpatialMixer defaults
                         (mambapy.pscan backend, d_state=8, expand=1x,
                         fused forward/backward pscan call).
  - `pscan (compiled)` : the same, wrapped in torch.compile.
  - `triton`           : same tuned config, fused Triton scan kernel backend
                         instead of mambapy.pscan (CUDA/ROCm only).
  - `attention`        : plain multi-head attention baseline.
  - `attention (compiled)`: same, compiled -- included so "compiled" isn't
                         an unfair advantage only given to our own model.

Run as: `python -m bee_training.chess_mamba.benchmark`
"""

import time

import torch
import torch.nn.functional as F
from torch import nn

from bee_training.chess_mamba.geometry import build_line_families
from bee_training.chess_mamba.spatial_mixer import KnightGraphMixer, SpatialMixer
from bee_training.chess_mamba.triton_scan import TRITON_AVAILABLE

D_MODEL = 128
BATCH = 64
N_ITERS = 15


class PlainAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        y, _ = self.mha(x, x, x)
        return y


# --------------------------------------------------------------------------
# Legacy (pre-pscan) scan, reconstructed only for a fair same-run comparison
# against the project's own history -- NOT used by the real model.
# --------------------------------------------------------------------------


class _LoopSelectiveSSM(nn.Module):
    def __init__(self, d_inner, d_state=16):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        dt_rank = max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x)
        dt_rank = x_dbl.shape[-1] - 2 * N
        dt, Bmat, Cmat = torch.split(x_dbl, [dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        if mask is not None:
            m = mask.unsqueeze(0).expand(B, L) if mask.dim() == 1 else mask
            dt = dt * m.unsqueeze(-1).to(dt.dtype)
        Abar = torch.exp(dt.unsqueeze(-1) * A)
        BX = dt.unsqueeze(-1) * Bmat.unsqueeze(2) * x.unsqueeze(-1)
        h = x.new_zeros(B, self.d_inner, N)
        hs = []
        for t in range(L):
            h = Abar[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)
        y = torch.einsum("bldn,bln->bld", hs, Cmat) + x * self.D
        if mask is not None:
            m = mask.unsqueeze(0).expand(B, L) if mask.dim() == 1 else mask
            y = y * m.unsqueeze(-1).to(y.dtype)
        return y


class _LoopMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2.0):
        super().__init__()
        d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.ssm = _LoopSelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x, mask=None):
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        y = self.ssm(x_in, mask=mask) * F.silu(z)
        return self.out_proj(y)


class _LoopDirectionalMamba(nn.Module):
    def __init__(self, d_model, idx, mask, d_state=16, expand=2.0):
        super().__init__()
        self.register_buffer("idx", idx)
        self.register_buffer("mask", mask)
        self.mamba_fwd = _LoopMambaBlock(d_model, d_state=d_state, expand=expand)
        self.mamba_bwd = _LoopMambaBlock(d_model, d_state=d_state, expand=expand)

    def forward(self, board_feats):
        B, _, D = board_feats.shape
        num_lines, max_len = self.idx.shape
        gathered = board_feats[:, self.idx.reshape(-1), :].view(B, num_lines, max_len, D)
        gathered = gathered * self.mask.unsqueeze(0).unsqueeze(-1)
        flat = gathered.reshape(B * num_lines, max_len, D)
        mask_tiled = self.mask.unsqueeze(0).expand(B, num_lines, max_len).reshape(-1, max_len)

        y_fwd = self.mamba_fwd(flat, mask=mask_tiled)
        y_bwd = self.mamba_bwd(flat.flip(dims=[1]), mask=mask_tiled.flip(dims=[-1])).flip(dims=[1])

        y = (y_fwd + y_bwd).view(B, num_lines, max_len, D) * self.mask.unsqueeze(0).unsqueeze(-1)
        out = board_feats.new_zeros(B, 64, D)
        out.index_add_(1, self.idx.reshape(-1), y.reshape(B, num_lines * max_len, D))
        return out


class LegacySpatialMixer(nn.Module):
    """Pre-optimization architecture: Python-loop scan, d_state=16, expand=2x,
    unfused forward/backward. For same-run benchmark comparison only."""

    def __init__(self, d_model, d_state=16, expand=2.0):
        super().__init__()
        families = build_line_families()
        self.rays = nn.ModuleDict({
            name: _LoopDirectionalMamba(d_model, idx, mask, d_state=d_state, expand=expand)
            for name, (idx, mask) in families.items()
        })
        self.knight = KnightGraphMixer(d_model)
        self.merge = nn.Linear(d_model * (len(families) + 1), d_model)

    def forward(self, board_feats):
        parts = [ray(board_feats) for ray in self.rays.values()]
        parts.append(self.knight(board_feats))
        return self.merge(torch.cat(parts, dim=-1))


# --------------------------------------------------------------------------
# Benchmark harness
# --------------------------------------------------------------------------


def bench(module, x, n_iters=N_ITERS):
    for _ in range(3):  # warmup
        y = module(x)
        y.sum().backward()
        module.zero_grad()
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y = module(x)
        y.sum().backward()
        module.zero_grad()
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters


def bench_compiled(module, x, n_iters=N_ITERS):
    """torch.compile has a real (one-time) compilation cost on the first
    call, on top of the usual warmup -- reported separately so it's not
    silently hidden inside (or silently inflating) the steady-state number."""
    compiled = torch.compile(module)
    t0 = time.perf_counter()
    y = compiled(x)
    y.sum().backward()
    compiled.zero_grad()
    if x.is_cuda:
        torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    steady_ms = bench(compiled, x, n_iters=n_iters) * 1000
    return compile_s, steady_ms


def run(device):
    torch.manual_seed(0)
    x = torch.randn(BATCH, 64, D_MODEL, device=device)
    results = []

    legacy = LegacySpatialMixer(D_MODEL, d_state=16, expand=2.0).to(device)
    results.append(("loop (pre-pscan, d_state=16 expand=2x, unfused)", bench(legacy, x) * 1000, None))

    tuned = SpatialMixer(D_MODEL, d_state=8, expand=1.0).to(device)
    results.append(("pscan (tuned: d_state=8 expand=1x, fused)", bench(tuned, x) * 1000, None))

    tuned_c = SpatialMixer(D_MODEL, d_state=8, expand=1.0).to(device)
    compile_s, steady_ms = bench_compiled(tuned_c, x)
    results.append(("pscan (tuned + torch.compile)", steady_ms, compile_s))

    if device == "cuda" and TRITON_AVAILABLE:
        triton_tuned = SpatialMixer(D_MODEL, d_state=8, expand=1.0, scan_backend="triton").to(device)
        results.append(("triton (tuned, fused)", bench(triton_tuned, x) * 1000, None))

        triton_tuned_c = SpatialMixer(D_MODEL, d_state=8, expand=1.0, scan_backend="triton").to(device)
        compile_s, steady_ms = bench_compiled(triton_tuned_c, x)
        results.append(("triton (tuned + torch.compile)", steady_ms, compile_s))

    attn = PlainAttention(D_MODEL).to(device)
    results.append(("attention (eager)", bench(attn, x) * 1000, None))

    attn_c = PlainAttention(D_MODEL).to(device)
    compile_s, steady_ms = bench_compiled(attn_c, x)
    results.append(("attention (torch.compile)", steady_ms, compile_s))

    attn_ms = next(ms for name, ms, _ in results if name == "attention (eager)")

    print(f"\n[{device}] batch={BATCH}, d_model={D_MODEL}, L=64")
    for name, ms, compile_s in results:
        warm = f"  (compile warmup: {compile_s:5.1f}s)" if compile_s is not None else ""
        print(f"  {name:48s} {ms:8.3f} ms/fwd+bwd   {ms/attn_ms:6.1f}x attention{warm}")
    print()


if __name__ == "__main__":
    run("cpu")
    if torch.cuda.is_available():
        run("cuda")
