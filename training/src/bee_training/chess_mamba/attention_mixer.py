"""
AttentionMixer: plain multi-head self-attention over the 64 squares, as a
drop-in alternative to SpatialMixer -- same (B, 64, D) -> (B, 64, D)
contract, so a ChessMambaBlock can use either interchangeably (see
`model.py`'s `HybridBlock`).

This is deliberately the *plain* attention baseline (no GAB/smolgen-style
dynamic positional bias) -- the point of this module is to be the cheap,
fast half of the hybrid, not to also chase GAB's own accuracy gains. A
learned board-conditioned bias is a natural follow-up if the hybrid's own
puzzle-accuracy ablation (plan Phase 6/7) suggests it's worth the extra
parameters and compute.
"""

from torch import nn


class AttentionMixer(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, board_feats):
        y, _ = self.mha(board_feats, board_feats, board_feats)
        return y
