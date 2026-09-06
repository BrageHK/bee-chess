"""Plots train/val loss curves from a checkpoint dir's `history.jsonl`.

Run as:
  python -m bee_training.chess_mamba.plot_history checkpoints/main-dawg-fast-transformer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train, val = [], []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if "train_loss" in record:
                train.append((record["step"], record["train_loss"]))
            if "val_loss" in record:
                val.append((record["step"], record["val_loss"]))
    return train, val


def plot(checkpoint_dir: Path, out_path: Path) -> None:
    train, val = load_history(checkpoint_dir / "history.jsonl")
    fig, ax = plt.subplots(figsize=(9, 5))
    if train:
        steps, losses = zip(*train)
        ax.plot(steps, losses, label="train loss", linewidth=1, alpha=0.8)
    if val:
        steps, losses = zip(*val)
        ax.plot(steps, losses, label="val loss", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"loss history -- {checkpoint_dir.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint_dir", type=Path)
    p.add_argument("--out", type=Path, default=None, help="defaults to <checkpoint_dir>/loss_plot.png")
    args = p.parse_args()
    out_path = args.out or (args.checkpoint_dir / "loss_plot.png")
    plot(args.checkpoint_dir, out_path)


if __name__ == "__main__":
    main()
