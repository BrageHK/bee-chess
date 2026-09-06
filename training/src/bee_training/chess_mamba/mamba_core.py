"""
Minimal selective SSM ("S6" / Mamba) block, with a pluggable scan backend
(see `SCAN_BACKENDS` below).

This is a from-scratch, readable reimplementation of the core Mamba
recurrence (Gu & Dao, 2023) -- NOT the official `mamba_ssm` CUDA package,
which requires custom CUDA/HIP kernels that do not build on RDNA3/gfx1100
ROCm (confirmed broken upstream on RX 7900-class cards as of 2026).
Instead, the scan is delegated to one of two in-process backends: the
default `mambapy.pscan.pscan` -- a Blelloch parallel-scan
`torch.autograd.Function` built entirely from plain tensor ops (add/mul),
so it needs no custom kernel and runs on any device -- or the opt-in
`triton_scan.triton_pscan`, a fused Triton kernel (CUDA/ROCm only) that's
faster but newer/less-battle-tested. Either way this turns the O(L)
sequential Python loop a hand-rolled version would need into O(log L)
tensor steps (pscan) or a single register-resident pass over L (triton).

Discretization uses the simple Euler approximation (Abar = exp(dt*A),
Bbar = dt*B) rather than the exact zero-order-hold used in the paper --
a standard simplification for readability that the official kernel does
slightly better, but that doesn't change the qualitative behaviour.

Supports an optional per-step `mask`: masked steps get dt forced to 0,
which makes Abar = 1 and Bbar = 0, i.e. the hidden state just passes
through unchanged and the step contributes nothing to the output. This
is what lets us pad variable-length diagonals up to a fixed length. This
masking is applied before/after the scan, so it's unaffected by which
scan implementation computes the recurrence.
"""

import torch
import torch.nn.functional as F
from mambapy.pscan import pscan as _mambapy_pscan
from torch import nn

from bee_training.chess_mamba.triton_scan import TRITON_AVAILABLE, triton_pscan

def _sequential_scan(Abar, BX):
    """h_t = Abar_t * h_{t-1} + BX_t, h_{-1} = 0, as a plain Python loop.

    O(L) sequential steps instead of pscan's O(log L), so it's slower for
    training at any real L -- but it has no custom autograd.Function and no
    in-place view mutation, both of which `mambapy.pscan` relies on and
    ONNX tracing does not reliably capture (a traced pscan call can come out
    of `torch.onnx.export` with its `A` input silently dropped, since the
    tracer loses the data dependency through pscan's chained in-place
    slice-mutations). At this project's scale (L<=8 per ray) the extra
    steps are free, so this is the backend `export_onnx.py` swaps in.

    Abar, BX: (B, L, D, N) -> returns hs: (B, L, D, N)
    """
    h = torch.zeros_like(Abar[:, 0])
    hs = []
    for t in range(Abar.shape[1]):
        h = Abar[:, t] * h + BX[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


# Pluggable scan backends, same (Abar, BX) -> hs contract either way (see
# each implementation's own docstring). "pscan" (mambapy, pure PyTorch) is
# the default -- it needs no custom kernel, so it's the safe choice on any
# device. "triton" is a fused kernel that streams over L instead of
# materializing the full (B,L,D,N) tensor pscan does; verified correct
# (matches the sequential reference and mambapy.pscan, including gradients)
# and ~1.2-1.4x faster end to end on this project's SpatialMixer on ROCm/
# gfx1100 -- but it's CUDA/ROCm-only (no CPU path) and a newer, less-used
# code path than mambapy, so it's opt-in rather than the default. "sequential"
# is for ONNX export only -- see `_sequential_scan`'s docstring.
SCAN_BACKENDS = {"pscan": _mambapy_pscan, "sequential": _sequential_scan}
if TRITON_AVAILABLE:
    SCAN_BACKENDS["triton"] = triton_pscan


def get_scan_fn(name: str):
    try:
        return SCAN_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown scan_backend {name!r}; available: {sorted(SCAN_BACKENDS)}"
        ) from None


class SelectiveSSM(nn.Module):
    """Core S6 recurrence, operating on an already-projected (B, L, D_inner) input."""

    def __init__(self, d_inner, d_state=16, dt_rank=None, scan_backend="pscan"):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self._scan = get_scan_fn(scan_backend)
        dt_rank = dt_rank or max(1, d_inner // 16)

        # input-dependent selection: project x -> (dt, B, C)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A is learned but NOT input-dependent (standard Mamba choice);
        # stored in log-space and negated so Abar = exp(-softplus-ish decay) stays stable
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # sensible dt bias init (as in the Mamba paper): steps mostly small
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

    def project(self, x, mask=None):
        """
        Everything up to (but not including) the scan itself: input-dependent
        dt/B/C, discretized into (Abar, BX, Cmat). Split out from `forward` so
        callers that want to batch several independent scans together (e.g.
        DirectionalMamba fusing its forward/backward passes) can concatenate
        Abar/BX along the batch dim and call `pscan` once instead of once per
        scan -- same math, fewer (and bigger) scan invocations.

        x    : (B, L, d_inner)
        mask : optional (L,) or (B, L) bool tensor, True = real step, False = padding
        returns (Abar, BX, Cmat) each (B, L, d_inner, d_state) / (B, L, d_state)
        """
        B, L, _D = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())  # (D, N), negative real => stable

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2N)
        dt_rank = x_dbl.shape[-1] - 2 * N
        dt, Bmat, Cmat = torch.split(x_dbl, [dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, D)

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, L)
            dt = dt * mask.unsqueeze(-1).to(dt.dtype)  # force dt=0 on padded steps

        # discretize: Abar_{b,l,d,n} = exp(dt_{b,l,d} * A_{d,n})
        Abar = torch.exp(dt.unsqueeze(-1) * A)                # (B, L, D, N)
        Bbar = dt.unsqueeze(-1) * Bmat.unsqueeze(2)            # (B, L, D, N)
        BX = Bbar * x.unsqueeze(-1)                            # (B, L, D, N)
        return Abar, BX, Cmat

    def combine(self, hs, Cmat, x, mask=None):
        """y_t = C_t . h_t + D*x_t, masked back to zero on padded steps."""
        y = torch.einsum("bldn,bln->bld", hs, Cmat)
        y = y + x * self.D
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(x.shape[0], x.shape[1])
            y = y * mask.unsqueeze(-1).to(y.dtype)
        return y

    def forward(self, x, mask=None):
        """
        x    : (B, L, d_inner)
        mask : optional (L,) or (B, L) bool tensor, True = real step, False = padding
        returns y: (B, L, d_inner)
        """
        Abar, BX, Cmat = self.project(x, mask=mask)
        # h_t = Abar_t * h_{t-1} + BX_t, h_{-1} = 0.
        hs = self._scan(Abar, BX)  # (B, L, D, N)
        return self.combine(hs, Cmat, x, mask=mask)


class MambaBlock(nn.Module):
    """Standard Mamba wrapper: in-proj -> SSM -> gate -> out-proj, with residual outside.

    `expand` defaults to 1x (d_inner=d_model), not the usual Mamba/LLM default
    of 2x: SpatialMixer's scans are line lengths <=8, and the "SSM is
    memory-bandwidth-bound at this scale" benchmark showed cost scales
    ~linearly with d_inner (and with d_state) -- there's no evidence a chess
    board's rank/file/diagonal scan needs 2x channel expansion to represent
    "propagate until blocked", so the cheaper default is used until an
    ablation shows otherwise (same reasoning ffn_mult=1.0 already uses
    elsewhere in this codebase, after LC0's own ablations found no benefit
    from the usual 4x FFN expansion for chess transformers).
    """

    def __init__(self, d_model, d_inner=None, d_state=16, expand=1.0, scan_backend="pscan"):
        super().__init__()
        d_inner = d_inner or int(expand * d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.ssm = SelectiveSSM(d_inner, d_state=d_state, scan_backend=scan_backend)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x, mask=None):
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        y = self.ssm(x_in, mask=mask)
        y = y * F.silu(z)
        return self.out_proj(y)


if __name__ == "__main__":
    torch.manual_seed(0)
    blk = MambaBlock(d_model=32)
    x = torch.randn(4, 8, 32)
    y = blk(x)
    print("no-mask:", y.shape)
    y.sum().backward()
    print("grad ok:", blk.in_proj.weight.grad is not None)

    # padding sanity check: padded steps should not affect earlier outputs
    mask = torch.tensor([True, True, True, False, False, False, False, False])
    x2 = x.clone()
    x2[:, 3:] = torch.randn(4, 5, 32) * 100  # garbage in the padded region
    torch.manual_seed(1)
    m = MambaBlock(d_model=32)
    y_a = m(x, mask=mask)
    y_b = m(x2, mask=mask)
    print("padding-invariant (should be ~0):", (y_a[:, :3] - y_b[:, :3]).abs().max().item())
