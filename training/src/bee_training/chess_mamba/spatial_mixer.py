"""
RaySSMMixer: the spatial "geometry" mixer for ChessMamba.

Idea: rook/bishop/queen attacks travel in straight lines and are blocked
by the first piece in the way. A selective-SSM scan along that exact line
is a natural, learnable stand-in for that ray-cast: the input-dependent
gate (dt) can learn to shrink to ~0 once it "sees" an occupied square,
so information stops propagating past a blocker -- something dot-product
attention (even with GAB) only ever approximates, since it does not have
an explicit notion of being blocked partway along a line.

Knight moves are not collinear with any of those four line families, so
they get their own mechanism: a fixed-adjacency local mixer (effectively
a tiny graph attention layer over the 64-square knight-move graph).

Layout: for each of the 4 line families we run ONE shared MambaBlock in
both directions (forward + reversed) over every line in that family, then
scatter-add the results back to their squares. The four families' outputs
are concatenated and projected back down to d_model.
"""

import torch
import torch.nn.functional as F
from torch import nn

from bee_training.chess_mamba.geometry import build_knight_adjacency, build_line_families
from bee_training.chess_mamba.mamba_core import MambaBlock, get_scan_fn


class DirectionalMamba(nn.Module):
    """One shared Mamba scan applied to every line of a single family, both ways.

    The forward and backward passes use separate learnable weights (they can
    learn different dynamics), but the actual scan call -- the expensive,
    memory-bandwidth-bound part -- doesn't care whose weights produced its
    (Abar, BX) inputs, since the scan is independent per batch element. So we
    run each direction's in_proj/x_proj/dt_proj separately (cheap), but
    concatenate their (Abar, BX) along the batch dim and call the scan ONCE
    for both directions instead of twice -- half the scan invocations, same
    math (see `test_fused_fwd_bwd_pscan_matches_two_separate_calls`).
    """

    def __init__(self, d_model, idx, mask, d_state=16, expand=1.0, scan_backend="pscan"):
        super().__init__()
        self.register_buffer("idx", idx)     # (num_lines, max_len)
        self.register_buffer("mask", mask)   # (num_lines, max_len)
        self.mamba_fwd = MambaBlock(d_model, d_state=d_state, expand=expand, scan_backend=scan_backend)
        self.mamba_bwd = MambaBlock(d_model, d_state=d_state, expand=expand, scan_backend=scan_backend)
        self._scan = get_scan_fn(scan_backend)

    def forward(self, board_feats):
        """board_feats: (B, 64, D) -> returns (B, 64, D), same-shape contribution."""
        B, _, D = board_feats.shape
        num_lines, max_len = self.idx.shape

        gathered = board_feats[:, self.idx.reshape(-1), :].view(B, num_lines, max_len, D)
        gathered = gathered * self.mask.unsqueeze(0).unsqueeze(-1)
        flat = gathered.reshape(B * num_lines, max_len, D)

        mask_tiled = self.mask.unsqueeze(0).expand(B, num_lines, max_len).reshape(-1, max_len)

        flat_fwd = flat
        flat_bwd = flat.flip(dims=[1])
        mask_fwd = mask_tiled
        mask_bwd = mask_tiled.flip(dims=[-1])

        # in_proj + SSM projection stay per-direction (separate weights, cheap)
        xz_fwd = self.mamba_fwd.in_proj(flat_fwd)
        x_fwd, z_fwd = xz_fwd.chunk(2, dim=-1)
        Abar_f, BX_f, Cmat_f = self.mamba_fwd.ssm.project(x_fwd, mask=mask_fwd)

        xz_bwd = self.mamba_bwd.in_proj(flat_bwd)
        x_bwd, z_bwd = xz_bwd.chunk(2, dim=-1)
        Abar_b, BX_b, Cmat_b = self.mamba_bwd.ssm.project(x_bwd, mask=mask_bwd)

        # one fused pscan call for both directions
        n_fwd = Abar_f.shape[0]
        Abar_cat = torch.cat([Abar_f, Abar_b], dim=0)
        BX_cat = torch.cat([BX_f, BX_b], dim=0)
        hs_cat = self._scan(Abar_cat, BX_cat)
        hs_f, hs_b = hs_cat[:n_fwd], hs_cat[n_fwd:]

        y_fwd = self.mamba_fwd.ssm.combine(hs_f, Cmat_f, x_fwd, mask=mask_fwd)
        y_fwd = self.mamba_fwd.out_proj(y_fwd * F.silu(z_fwd))

        y_bwd = self.mamba_bwd.ssm.combine(hs_b, Cmat_b, x_bwd, mask=mask_bwd)
        y_bwd = self.mamba_bwd.out_proj(y_bwd * F.silu(z_bwd))
        y_bwd = y_bwd.flip(dims=[1])

        y = (y_fwd + y_bwd).view(B, num_lines, max_len, D)
        y = y * self.mask.unsqueeze(0).unsqueeze(-1)

        out = board_feats.new_zeros(B, 64, D)
        idx_flat = self.idx.reshape(-1)          # (num_lines*max_len,)
        y_flat = y.reshape(B, num_lines * max_len, D)
        out.index_add_(1, idx_flat, y_flat)
        return out


class KnightGraphMixer(nn.Module):
    """Fixed-adjacency local mixer over the knight-move graph (avg degree 5.25)."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        idx, mask = build_knight_adjacency()
        self.register_buffer("idx", idx)    # (64, 8)
        self.register_buffer("mask", mask)  # (64, 8)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, board_feats):
        B, S, D = board_feats.shape  # S = 64
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(board_feats).view(B, S, H, Dh)
        k_all = self.k_proj(board_feats).view(B, S, H, Dh)
        v_all = self.v_proj(board_feats).view(B, S, H, Dh)

        # gather each square's knight-neighbor keys/values: (B, S, 8, H, Dh)
        neigh_idx = self.idx.view(1, S, 8, 1, 1).expand(B, S, 8, H, Dh)
        k = torch.gather(
            k_all.unsqueeze(1).expand(B, S, S, H, Dh), 2, neigh_idx
        )
        v = torch.gather(
            v_all.unsqueeze(1).expand(B, S, S, H, Dh), 2, neigh_idx
        )

        attn = torch.einsum("bshd,bsnhd->bsnh", q, k) / (Dh ** 0.5)  # (B,S,8,H)
        neigh_mask = self.mask.view(1, S, 8, 1)
        attn = attn.masked_fill(~neigh_mask, float("-inf"))
        weights = torch.softmax(attn, dim=2)
        weights = torch.nan_to_num(weights, nan=0.0)  # squares with 0 valid neighbors (none here)

        out = torch.einsum("bsnh,bsnhd->bshd", weights, v).reshape(B, S, D)
        return self.out_proj(out)


class SpatialMixer(nn.Module):
    """Combines all 4 ray directions + the knight graph into one geometry-aware mixer."""

    def __init__(self, d_model, d_state=8, expand=1.0, scan_backend="pscan"):
        super().__init__()
        families = build_line_families()
        self.rays = nn.ModuleDict({
            name: DirectionalMamba(d_model, idx, mask, d_state=d_state, expand=expand,
                                    scan_backend=scan_backend)
            for name, (idx, mask) in families.items()
        })
        self.knight = KnightGraphMixer(d_model)
        self.merge = nn.Linear(d_model * (len(families) + 1), d_model)

    def forward(self, board_feats):
        parts = [ray(board_feats) for ray in self.rays.values()]
        parts.append(self.knight(board_feats))
        return self.merge(torch.cat(parts, dim=-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    mixer = SpatialMixer(d_model=32)
    x = torch.randn(4, 64, 32)
    y = mixer(x)
    print("SpatialMixer output:", y.shape)
    y.sum().backward()
    print("grads flow:", all(p.grad is not None for p in mixer.parameters() if p.requires_grad))
