"""Honest speed check: at L=64 (one board), is the SSM mixer actually
faster than plain attention? (Spoiler: it shouldn't be -- the pitch for
this architecture is inductive bias, not speed, at this sequence length.)

Runs on CPU and, if available, on the GPU (ROCm shows up as `cuda`).
"""

import time

import torch
from torch import nn

from bee_training.chess_mamba.spatial_mixer import SpatialMixer


class PlainAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        y, _ = self.mha(x, x, x)
        return y


def bench(module, x, n_iters=20):
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


def run(device, n_iters):
    torch.manual_seed(0)
    d_model, B = 128, 8
    x = torch.randn(B, 64, d_model, device=device)

    ssm_mixer = SpatialMixer(d_model).to(device)
    attn_mixer = PlainAttention(d_model).to(device)

    t_ssm = bench(ssm_mixer, x, n_iters=n_iters)
    t_attn = bench(attn_mixer, x, n_iters=n_iters)

    print(f"[{device}] SpatialMixer (SSM, ours): {t_ssm*1000:7.2f} ms / fwd+bwd  "
          f"({sum(p.numel() for p in ssm_mixer.parameters()):,} params)")
    print(f"[{device}] Plain multi-head attn:    {t_attn*1000:7.2f} ms / fwd+bwd  "
          f"({sum(p.numel() for p in attn_mixer.parameters()):,} params)")
    print(f"[{device}] SSM is {t_ssm/t_attn:.1f}x the wall-clock time of plain attention at L=64.\n")


if __name__ == "__main__":
    run("cpu", n_iters=5)
    if torch.cuda.is_available():
        run("cuda", n_iters=20)
