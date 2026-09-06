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

`forward` computes standard scaled-dot-product attention with plain
matmul/softmax ops rather than calling `self.mha(...)` (which dispatches to
`F.multi_head_attention_forward`): that internal path traces to ONNX as a
5-D unsqueeze/permute/squeeze dance (to stay generic over batch_first,
key_padding_mask, separate-Q/K/V weights, etc., none of which this module
ever uses) that both onnxruntime-web and burn-onnx's ONNX-to-Rust codegen
choke on for different reasons (see export_onnx.py and mz-web's ChessMamba
integration). `self.mha` is kept only as the parameter container (its
`in_proj_weight`/`in_proj_bias`/`out_proj` give this the exact same
state_dict keys and shapes as before), so already-trained checkpoints still
load; only the forward computation changed, and it's the same math (no
attn_mask/key_padding_mask/bias_k/bias_v -- this project never sets any of
those), so trained weights behave identically.
"""

import torch
import torch.nn.functional as F
from torch import nn


class AttentionMixer(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

    def forward(self, board_feats):
        B, L, D = board_feats.shape
        H, Dh = self.n_heads, self.d_head

        qkv = F.linear(board_feats, self.mha.in_proj_weight, self.mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, Dh)

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.mha.out_proj(out)
