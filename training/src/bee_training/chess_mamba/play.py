"""
UCI-speaking wrapper around a trained `ChessMamba` checkpoint.

`ChessMamba` is searchless (see model.py's docstring): there is no
alpha-beta or MCTS here, just one forward pass per `go`, with the policy
head's (from, to) logits picked directly over the position's legal moves.
This lets a checkpoint be played against like any other UCI engine --
`bridge/server.py` spawns this alongside Stockfish and Bee so the frontend
can talk to it the same way (see ADR 0001: this is a dev/visualization
tool, not the v1 competition engine).

Run as:
  python -m bee_training.chess_mamba.play --checkpoint checkpoints/main-dawg/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import torch

from bee_training.chess_mamba.encode import encode_fen
from bee_training.chess_mamba.train import TrainConfig, build_model

ENGINE_NAME = "Bee-Mamba"
ENGINE_AUTHOR = "bee-chess"


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**ckpt["config"])
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def choose_move(model: torch.nn.Module, board: chess.Board, device: str) -> chess.Move | None:
    """Picks the legal move whose (from, to) squares score highest under
    the policy head. Returns None if the position has no legal move.

    Underpromotions are pruned to queen promotion: `FromToPolicyHead`
    only scores square pairs, so it can't tell a queen promotion apart
    from an underpromotion to the same square -- and queen is virtually
    always the right choice anyway.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    planes = encode_fen(board.fen()).unsqueeze(0).to(device)
    policy_logits, _ = model(planes)
    policy_logits = policy_logits[0]  # (64, 64)

    best_move, best_score = None, float("-inf")
    for move in legal_moves:
        score = policy_logits[move.from_square, move.to_square].item()
        if move.promotion is not None and move.promotion != chess.QUEEN:
            score = float("-inf")
        if score > best_score:
            best_move, best_score = move, score
    return best_move


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


def run(checkpoint_path: Path, device: str, in_stream=sys.stdin, out_stream=sys.stdout) -> None:
    model = load_model(checkpoint_path, device)
    board = chess.Board()

    def send(line: str) -> None:
        print(line, file=out_stream, flush=True)

    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            send(f"id name {ENGINE_NAME}")
            send(f"id author {ENGINE_AUTHOR}")
            send("uciok")
        elif command == "isready":
            send("readyok")
        elif command == "ucinewgame":
            board.reset()
        elif command == "position":
            _apply_position_command(board, tokens[1:])
        elif command == "go":
            move = choose_move(model, board, device)
            send(f"bestmove {move.uci() if move else '(none)'}")
        elif command == "quit":
            return
        # setoption and anything else this engine doesn't support are
        # silently ignored -- it has no tunable options.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/main-dawg/latest.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.checkpoint, args.device)


if __name__ == "__main__":
    main()
