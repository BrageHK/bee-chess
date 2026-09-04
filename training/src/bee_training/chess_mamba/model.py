"""
ChessMamba: a Mamba/SSM-based searchless chess model.

Board representation follows the square-token recipe from Chessformer /
Leela's BT-series (64 tokens, one-hot piece per square, optional history
planes) rather than the original Grandmaster-Level paper's FEN-character
tokenization -- square tokens are what make a geometry-aware mixer like
this one possible in the first place.

Where Chessformer replaces RoPE/absolute position encodings with a
learned, board-state-conditioned attention bias (GAB), this model
replaces the *entire* geometric mixing step (GAB + dot-product attention)
with directional selective-SSM scans along the rook/bishop/queen lines,
plus a small fixed-adjacency mixer for knight moves. See spatial_mixer.py
for the reasoning.

Two output heads, matching what's already been validated to work well:
  - value head: mean-pool -> MLP -> K-way classification (HL-Gauss style)
  - policy head: attention-based "from-square, to-square" head (Chessformer)
"""

import torch
import torch.nn.functional as F
from torch import nn

from bee_training.chess_mamba.attention_mixer import AttentionMixer
from bee_training.chess_mamba.spatial_mixer import SpatialMixer

N_PIECE_TYPES = 12  # 6 piece types x 2 colors
N_SQUARES = 64


def hybrid_layer_types(n_layers, n_ssm=2):
    """Recommended hybrid layout: `n_ssm` SpatialMixer (SSM ray-scan) layers
    first, plain AttentionMixer for the rest.

    Rationale (see benchmark.py / README for the numbers): the SSM ray
    scan is 20-40x slower than plain attention at board scale (L=64), so
    running it in every layer is expensive for a bet that's about
    inductive bias, not speed. Putting it only in the first couple of
    layers keeps the "explicit blocker-stop" mechanism where it plausibly
    matters most -- extracting raw sliding-piece line-of-sight structure
    close to the input -- while the remaining, cheaper attention layers
    handle higher-level mixing. This is the same pattern real hybrid
    SSM/attention models use (Jamba, Griffin/Hawk, Zamba): interleave a
    few attention layers among many SSM layers (or vice versa), not an
    all-or-nothing choice between the two.
    """
    if not 0 <= n_ssm <= n_layers:
        raise ValueError(f"n_ssm={n_ssm} must be between 0 and n_layers={n_layers}")
    return ["ssm"] * n_ssm + ["attn"] * (n_layers - n_ssm)


class ChessMambaBlock(nn.Module):
    """One residual block: mixer (SpatialMixer or AttentionMixer) + FFN.

    `mixer_type` picks which geometry/context mixer this block uses -- see
    `hybrid_layer_types` for why you'd want to mix the two across a model's
    layers rather than use just one throughout.
    """

    def __init__(self, d_model, mixer_type="ssm", d_state=8, expand=1.0, scan_backend="pscan",
                 n_heads=8, ffn_mult=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if mixer_type == "ssm":
            self.mixer = SpatialMixer(d_model, d_state=d_state, expand=expand, scan_backend=scan_backend)
        elif mixer_type == "attn":
            self.mixer = AttentionMixer(d_model, n_heads=n_heads)
        else:
            raise ValueError(f"unknown mixer_type {mixer_type!r}; expected 'ssm' or 'attn'")
        self.norm2 = nn.LayerNorm(d_model)
        d_ffn = int(d_model * ffn_mult)
        # LC0's ablations found little benefit from the usual 4x FFN expansion
        # for chess transformers; we default to 1x and let you raise ffn_mult.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FromToPolicyHead(nn.Module):
    """Attention-style source/destination head (Chessformer), backbone-agnostic."""

    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, board_feats):
        q = self.q_proj(board_feats)  # "from" queries, (B, 64, D)
        k = self.k_proj(board_feats)  # "to" keys,      (B, 64, D)
        logits = torch.einsum("bfd,btd->bft", q, k) * self.scale  # (B, 64, 64)
        return logits  # flatten to (B, 4096) for a 64x64 move-logit matrix


class ValueHead(nn.Module):
    """Mean-pool -> MLP -> K bins. Train with HL-Gauss / categorical cross-entropy,
    not MSE regression (Farebrother et al., 2024; already used in the original
    Grandmaster-Level Chess paper's value head)."""

    def __init__(self, d_model, n_bins=128, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, board_feats):
        pooled = self.norm(board_feats.mean(dim=1))
        return self.mlp(pooled)  # (B, n_bins) logits


class ChessMamba(nn.Module):
    """
    `layer_types`: optional list of `n_layers` entries, each `"ssm"` or
    `"attn"`, picking each block's mixer. Defaults to all-`"ssm"` (the
    original architecture) for backwards compatibility -- pass
    `hybrid_layer_types(n_layers)` (or your own mix) to build the hybrid.
    """

    def __init__(self, d_model=256, n_layers=8, d_state=8, expand=1.0, scan_backend="pscan",
                 n_heads=8, layer_types=None, n_history=7, n_value_bins=128, ffn_mult=1.0):
        super().__init__()
        if layer_types is None:
            layer_types = ["ssm"] * n_layers
        if len(layer_types) != n_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries, expected n_layers={n_layers}")

        in_dim = N_PIECE_TYPES * (n_history + 1) + 8  # +8 for castling/ep/rule50/etc.
        self.embed = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([
            ChessMambaBlock(d_model, mixer_type=mixer_type, d_state=d_state, expand=expand,
                             scan_backend=scan_backend, n_heads=n_heads, ffn_mult=ffn_mult)
            for mixer_type in layer_types
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.policy_head = FromToPolicyHead(d_model)
        self.value_head = ValueHead(d_model, n_bins=n_value_bins)

    def forward(self, board_planes):
        """board_planes: (B, 64, in_dim) -- one-hot piece/history/aux planes per square."""
        x = self.embed(board_planes)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        policy_logits = self.policy_head(x)   # (B, 64, 64)
        value_logits = self.value_head(x)     # (B, n_value_bins)
        return policy_logits, value_logits


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ChessMamba(d_model=64, n_layers=2, d_state=8, n_history=7)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    B = 3
    in_dim = N_PIECE_TYPES * 8 + 8
    dummy = torch.randn(B, N_SQUARES, in_dim)
    policy_logits, value_logits = model(dummy)
    print("policy_logits:", policy_logits.shape)  # (B, 64, 64)
    print("value_logits:", value_logits.shape)     # (B, n_bins)

    # smoke-test a training step end to end
    target_move = torch.randint(0, 64 * 64, (B,))
    target_bin = torch.randint(0, 128, (B,))
    loss = F.cross_entropy(policy_logits.reshape(B, -1), target_move) \
        + F.cross_entropy(value_logits, target_bin)
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"loss: {loss.item():.3f}  params with grad: {n_with_grad}/{n_total}")
