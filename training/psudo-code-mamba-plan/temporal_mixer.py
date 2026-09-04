"""
TemporalHistoryMamba: a second, complementary use of Mamba for chess.

SpatialMixer (spatial_mixer.py) uses Mamba WITHIN a single position, along
board geometry. This module instead uses Mamba ACROSS positions, along the
game's move history -- a genuinely long, genuinely causal sequence, which
is exactly the regime Mamba was built for (unlike the within-position
case, where 64 tokens is already cheap for plain attention).

Chessformer/BT4 condition on history by concatenating the last n=7 raw
board planes as extra input channels -- a fixed window, recomputed from
scratch every ply. This module instead keeps a single running hidden
state that gets updated by one Mamba step per new ply, so:
  - history length is unbounded (not fixed at n=7), and
  - updating for a new move is O(1), not O(n) -- relevant for a live
    engine that evaluates many positions along one game, or for modeling
    a human player's full game so far rather than a fixed recent window.
"""

import torch
import torch.nn as nn

from mamba_core import MambaBlock


class TemporalHistoryMamba(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, position_summaries):
        """
        position_summaries: (B, T, D) -- one summary vector per ply so far
        (e.g. mean-pooled ChessMamba board features), earliest ply first.
        Returns (B, T, D): a running game-context vector at every ply;
        in live play you only need the last one, [:, -1].
        """
        return self.norm(self.mamba(position_summaries))

    @torch.no_grad()
    def step(self, new_summary, state=None):
        """
        Incremental single-ply update for live play. This is a reference
        implementation using the same batched module for clarity; a
        production version would carry (h, conv-state) explicitly instead
        of re-deriving it, but the O(1)-per-move contract is the point --
        contrast with re-concatenating and reprocessing a fixed window.
        """
        raise NotImplementedError(
            "Reference version only demonstrates the O(1)-per-ply idea via "
            "forward(); a production port should expose Mamba's recurrent "
            "single-step API (as mamba_ssm.Mamba.step does) instead of "
            "replaying history through forward() each time."
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model = 64
    mixer = TemporalHistoryMamba(d_model)
    B, T = 2, 40  # a 40-ply game so far
    summaries = torch.randn(B, T, d_model)
    out = mixer(summaries)
    print("TemporalHistoryMamba output:", out.shape)
    out.sum().backward()
    print("grads flow:", all(p.grad is not None for p in mixer.parameters()))
