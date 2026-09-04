"""
Minimal, pure-PyTorch selective SSM ("S6" / Mamba) block.

This is a from-scratch, readable reimplementation of the core Mamba
recurrence (Gu & Dao, 2023) -- NOT the official `mamba_ssm` CUDA package.
It's intentionally simple because every sequence we scan in this project
(a chess rank/file/diagonal, or a game's move history) is short (<= 64
steps), so a plain Python-level sequential scan is fast enough and there's
no need for the parallel-scan CUDA kernel that makes the official package
fast on sequences of thousands of tokens.

Discretization uses the simple Euler approximation (Abar = exp(dt*A),
Bbar = dt*B) rather than the exact zero-order-hold used in the paper --
a standard simplification for readability that the official kernel does
slightly better, but that doesn't change the qualitative behaviour.

Supports an optional per-step `mask`: masked steps get dt forced to 0,
which makes Abar = 1 and Bbar = 0, i.e. the hidden state just passes
through unchanged and the step contributes nothing to the output. This
is what lets us pad variable-length diagonals up to a fixed length.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Core S6 recurrence, operating on an already-projected (B, L, D_inner) input."""

    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
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

    def forward(self, x, mask=None):
        """
        x    : (B, L, d_inner)
        mask : optional (L,) or (B, L) bool tensor, True = real step, False = padding
        returns y: (B, L, d_inner)
        """
        B, L, D = x.shape
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

        h = x.new_zeros(B, D, N)
        ys = []
        for t in range(L):
            h = Abar[:, t] * h + Bbar[:, t] * x[:, t].unsqueeze(-1)   # (B, D, N)
            y_t = torch.einsum("bdn,bn->bd", h, Cmat[:, t])           # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + x * self.D
        if mask is not None:
            y = y * mask.unsqueeze(-1).to(y.dtype)
        return y


class MambaBlock(nn.Module):
    """Standard Mamba wrapper: in-proj -> SSM -> gate -> out-proj, with residual outside."""

    def __init__(self, d_model, d_inner=None, d_state=16):
        super().__init__()
        d_inner = d_inner or 2 * d_model
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.ssm = SelectiveSSM(d_inner, d_state=d_state)
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
    y_real = MambaBlock(32)
    torch.manual_seed(1)
    m = MambaBlock(d_model=32)
    y_a = m(x, mask=mask)
    y_b = m(x2, mask=mask)
    print("padding-invariant (should be ~0):", (y_a[:, :3] - y_b[:, :3]).abs().max().item())
