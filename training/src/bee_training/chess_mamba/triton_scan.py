"""
Fused Triton selective-scan kernel: a drop-in alternative backend for
`mambapy.pscan.pscan`, aimed at RDNA3/gfx1100 ROCm (RX 7900-class cards).

Same shape contract as `mambapy.pscan.pscan`:
    triton_pscan(A, X) -> H
    A, X, H : (B, L, D, N)
    H[t] = A[t] * H[t-1] + X[t],  H[-1] = 0

Why this exists: `mambapy.pscan` is a Blelloch parallel scan built from
plain tensor ops. It's correct and needs no custom kernel, but for our
workload (L <= 8 always -- board rank/file/diagonal lines are never
longer than 8 squares) it pays for parallelism we don't need: it clones
its inputs, pads to the next power of two, and runs several up-sweep/
down-sweep passes, each a separate elementwise kernel launch materializing
a full (B, L, D, N) intermediate. Measured on this project's SpatialMixer,
that made it memory-bandwidth-bound rather than launch-overhead-bound --
cost scaled ~linearly with D*N, and the SSM-vs-attention slowdown ratio
got *worse* with bigger batch (not better, as launch-overhead-bound code
would), plus it OOM'd at B=1024 on a 24GB card.

Since L is tiny and known at trace time, a much simpler kernel is
possible: one Triton program per block of (b, d, n) "channels", each
holding its scalar hidden state in a register and looping over L
(Python-unrolled at trace time, since L is `tl.constexpr`), doing exactly
one load of A and X and one store of H per step -- the same memory
traffic a fused CUDA kernel (e.g. the official `mamba_ssm`) would use,
and no more. This is what the official kernel does and can't do here
because it doesn't build on gfx1100 ROCm; this is the from-scratch
equivalent using Triton, whose AMD backend does work on this GPU
(confirmed elsewhere in this project via `torch.compile`).

The recurrence has no interaction across d or n (A/X are already fully
elementwise per (d, n), with the "state matrix" structure baked into
Abar/BX before this function ever sees them), so it's valid to flatten
(D, N) into one channel axis R = D*N and parallelize over (B, R).

Backward pass: standard linear-recurrence adjoint. Let g_t = dL/dH_t
(the incoming cotangent). The *total* gradient reaching H_t (accounting
for how H_t feeds forward into every later step) is

    G_{L-1} = g_{L-1}
    G_t     = g_t + A_{t+1} * G_{t+1}          for t = L-2 .. 0

which is itself the same kind of linear scan, run backward in time with
A shifted by one step (and the coefficient at the last step unused,
since G_{L-1}'s "previous" state is 0). Then

    dL/dX_t = G_t
    dL/dA_t = G_t * H_{t-1}                     (H_{-1} = 0)

so both the forward and backward passes reduce to one call each of the
same kernel (forward direction, then reverse direction with a shifted
`A`), reusing this module's `_run_scan` for both.
"""

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _scan_kernel(
        a_ptr, x_ptr, out_ptr,
        stride_b, stride_l,
        R,
        L: tl.constexpr,
        BLOCK_R: tl.constexpr,
        REVERSE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)

        r_offsets = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offsets < R
        base = pid_b * stride_b + r_offsets

        h = tl.zeros((BLOCK_R,), dtype=tl.float32)

        # L is a compile-time constant (always <= 8 for this project), so
        # these Python-level loops are unrolled at trace time -- no runtime
        # loop overhead, h stays register-resident throughout. (Triton's
        # frontend wants the `for` written directly inside each constexpr
        # branch rather than looping over a `range` object assigned outside
        # an `if`/`else`, hence the duplication instead of a shared range.)
        if REVERSE:
            for l in range(L - 1, -1, -1):
                offset = base + l * stride_l
                a = tl.load(a_ptr + offset, mask=r_mask, other=0.0)
                x = tl.load(x_ptr + offset, mask=r_mask, other=0.0)
                h = a * h + x
                tl.store(out_ptr + offset, h, mask=r_mask)
        else:
            for l in range(L):
                offset = base + l * stride_l
                a = tl.load(a_ptr + offset, mask=r_mask, other=0.0)
                x = tl.load(x_ptr + offset, mask=r_mask, other=0.0)
                h = a * h + x
                tl.store(out_ptr + offset, h, mask=r_mask)


def _run_scan(A: torch.Tensor, X: torch.Tensor, reverse: bool, block_r: int = 256) -> torch.Tensor:
    """A, X: (B, L, D, N) contiguous float32 CUDA tensors -> H: (B, L, D, N)."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed; triton_scan backend unavailable")
    if not A.is_cuda:
        raise RuntimeError("triton_scan requires CUDA/ROCm tensors")

    B, L, D, N = A.shape
    R = D * N
    A_flat = A.contiguous().view(B, L, R)
    X_flat = X.contiguous().view(B, L, R)
    out = torch.empty_like(A_flat)

    grid = (B, triton.cdiv(R, block_r))
    _scan_kernel[grid](
        A_flat, X_flat, out,
        A_flat.stride(0), A_flat.stride(1),
        R,
        L=L,
        BLOCK_R=block_r,
        REVERSE=reverse,
    )
    return out.view(B, L, D, N)


class _TritonScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        hs = _run_scan(A, X, reverse=False)
        ctx.save_for_backward(A, hs)
        return hs

    @staticmethod
    def backward(ctx, grad_hs):
        A, hs = ctx.saved_tensors
        grad_hs = grad_hs.contiguous()

        # shifted_A[t] = A[t+1] for t < L-1, else 0 (unused: multiplied by
        # the reverse scan's initial h=0 at its first step).
        shifted_A = torch.zeros_like(A)
        shifted_A[:, :-1] = A[:, 1:]

        G = _run_scan(shifted_A, grad_hs, reverse=True)  # G_t = dL/dH_t (total)
        grad_X = G

        # shifted_hs[t] = hs[t-1] for t > 0, else 0 (H_{-1} = 0)
        shifted_hs = torch.zeros_like(hs)
        shifted_hs[:, 1:] = hs[:, :-1]
        grad_A = G * shifted_hs

        return grad_A, grad_X


def triton_pscan(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for `mambapy.pscan.pscan(A, X)`. See module
    docstring for the recurrence and shape contract."""
    return _TritonScan.apply(A, X)
