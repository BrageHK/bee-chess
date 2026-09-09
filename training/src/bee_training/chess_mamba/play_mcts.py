"""
UCI-speaking wrapper around a trained `ChessMamba` checkpoint that searches
instead of just argmaxing the policy head (see `play.py`'s docstring for
that simpler, searchless player).

Runs an lc0-style PUCT MCTS -- `mamba_mcts_native` (a Rust/PyO3 extension;
see its README for the full design and benchmark history) owns the
tree/PUCT/FPU/virtual-loss logic, chess move generation, and an
NN-evaluation cache; this module owns loading the checkpoint and running
NN inference through it, so PyTorch's own backend selection (CPU/MPS/CUDA)
picks whatever's fastest on the host. Batched: `mamba_mcts_native` gathers
`--batch-size` pending leaves per wave and this module evaluates all of
them in one forward pass, not one leaf at a time -- the thing that lets a
GPU backend actually win (see `mamba_mcts_native`'s README for measured
numbers; batch=1 was consistently *slower* than CPU on every GPU backend
tried during development).

`--scan-backend sequential` (the default here) overrides the checkpoint's
own trained config (`main-dawg` was trained with "pscan"). Measured
directly on Apple Silicon's MPS backend: "sequential" -- a plain per-step
loop over this model's tiny per-ray sequence length (<=8) -- is faster
than "pscan"'s O(log L) parallel-scan indexing, whose overhead doesn't pay
off at such a small L. (pscan generally wins on CUDA, where kernel-launch
cost is cheap and parallelism scales -- pass `--scan-backend pscan` there.)

Run as:
  python -m bee_training.chess_mamba.play_mcts --checkpoint checkpoints/main-dawg/latest.pt
Or, once installed (see pyproject.toml's `[project.scripts]`):
  bee-mamba-mcts-uci
Point any UCI harness at either invocation -- a GUI, cutechess-cli, or
lichess-bot's `homemade`/`uci` engine config for playing on Lichess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import mamba_mcts_native
import numpy as np
import torch

from bee_training.chess_mamba.train import TrainConfig, build_model

ENGINE_NAME = "Bee-Mamba-MCTS"
ENGINE_AUTHOR = "bee-chess"

DEFAULT_SIMULATIONS = 800
DEFAULT_BATCH_SIZE = 32  # see mamba_mcts_native's README for why this is the measured sweet spot


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: Path, device: str, scan_backend: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**{**ckpt["config"], "scan_backend": scan_backend})
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


class BatchedEvaluator:
    """The `evaluate_batch` callback `mamba_mcts_native.search` calls back
    into: takes the (n, 64, IN_DIM) plane batch it already encoded in Rust
    (no FEN re-parsing here -- see that crate's README for why that matters)
    and runs it through the network, returning plain numpy arrays."""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device

    @torch.inference_mode()  # stricter than no_grad(): also skips view-tracking/version-counter bookkeeping
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(planes).to(self.device)
        policy_logits, value_logits = self.model(x)
        policy_flat = policy_logits.reshape(policy_logits.shape[0], -1)
        return policy_flat.to("cpu").numpy(), value_logits.to("cpu").numpy()


def choose_move(
    evaluator: BatchedEvaluator, board: chess.Board, simulations: int, batch_size: int
) -> chess.Move | None:
    best_uci = mamba_mcts_native.search(
        board.fen(), evaluator, simulations=simulations, batch_size=batch_size
    )
    return chess.Move.from_uci(best_uci) if best_uci else None


def _apply_position_command(board: chess.Board, tokens: list[str]) -> None:
    """Handles a UCI `position` command's tokens (after the leading
    `position` itself), e.g. `startpos moves e2e4 e7e5` or
    `fen <6 fields> moves ...`."""
    if tokens[0] == "startpos":
        board.reset()
        rest = tokens[1:]
    else:
        assert tokens[0] == "fen"
        board.set_fen(" ".join(tokens[1:7]))
        rest = tokens[7:]
    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            board.push_uci(uci)


def run(
    checkpoint_path: Path,
    device: str,
    scan_backend: str,
    simulations: int,
    batch_size: int,
    in_stream=sys.stdin,
    out_stream=sys.stdout,
) -> None:
    model = load_model(checkpoint_path, device, scan_backend)
    evaluator = BatchedEvaluator(model, device)
    board = chess.Board()

    def send(line: str) -> None:
        print(line, file=out_stream, flush=True)

    def handle_setoption(rest: list[str]) -> None:
        nonlocal simulations, batch_size
        if len(rest) < 4 or rest[0] != "name" or rest[2] != "value":
            return
        name, value = rest[1], rest[3]
        if name == "Simulations":
            simulations = int(value)
        elif name == "BatchSize":
            batch_size = int(value)

    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            send(f"id name {ENGINE_NAME}")
            send(f"id author {ENGINE_AUTHOR}")
            send(
                f"option name Simulations type spin default {DEFAULT_SIMULATIONS} min 1 max 100000"
            )
            send(f"option name BatchSize type spin default {DEFAULT_BATCH_SIZE} min 1 max 512")
            send("uciok")
        elif command == "isready":
            send("readyok")
        elif command == "ucinewgame":
            board.reset()
        elif command == "setoption":
            # `setoption name Simulations value 400` etc.
            handle_setoption(tokens[1:])
        elif command == "position":
            _apply_position_command(board, tokens[1:])
        elif command == "go":
            move = choose_move(evaluator, board, simulations, batch_size)
            send(f"bestmove {move.uci() if move else '(none)'}")
        elif command == "quit":
            return
        # Any other command this engine doesn't support is silently
        # ignored, same as play.py.


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/main-dawg/latest.pt"))
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--scan-backend", default="sequential", choices=["sequential", "pscan"])
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.scan_backend, args.simulations, args.batch_size)


if __name__ == "__main__":
    main()
