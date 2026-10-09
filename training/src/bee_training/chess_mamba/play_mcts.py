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
import time
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

# Time management (used whenever a `go` carries clock info -- wtime/btime or
# movetime -- instead of the fixed Simulations option; see `sims_for_go`).
# Same shape as most UCI engines' simple time managers: budget one move as a
# slice of the remaining clock plus most of the increment, capped so no
# single move can eat too much of the clock, with an extra clamp once time
# gets critically short so the engine doesn't flag.
ASSUMED_TOTAL_MOVES = 40  # moves-left estimate when the GUI doesn't send movestogo
MIN_MOVES_LEFT = 10  # floor for that estimate late in the game
INCREMENT_WEIGHT = 0.8  # how much of the increment to bank on top of the slice
MAX_CLOCK_FRACTION = 0.5  # never plan to spend more than half the remaining clock on one move
LOW_TIME_THRESHOLD_MS = 1000  # below this, throttle further to avoid flagging
LOW_TIME_FRACTION = 0.3
MIN_BUDGET_MS = 50.0
MIN_SIMULATIONS = 16
# Safety ceiling only -- not meant to bind in practice. Measured nps in real
# games is ~25-40k/s; the worst case this bot will accept (challenge.max_base
# 1800s + max_increment 20s, early-game moves_left=39) computes a budget of
# ~60s, i.e. ~2M sims, and the deepest late-game case (moves_left floor of
# MIN_MOVES_LEFT with a big remaining clock) can reach ~4-5M. This used to be
# 100_000, which was far below what real per-move budgets computed to (e.g.
# ~2.4s of actual search per move in a 3+1 blitz game against a ~6-7s
# budget) -- the bot was flooring out on this cap instead of spending the
# time it had.
MAX_SIMULATIONS = 10_000_000


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


def _game_history_fens(board: chess.Board) -> list[str]:
    """Every position the real game has passed through so far, oldest
    first, *not* including `board`'s own current position -- what the
    native search needs to tell a genuine threefold-repetition/fifty-move
    draw apart from a position its NN-eval cache has merely seen before
    (see `mamba_mcts_native`'s README/`lib.rs` for why that distinction
    matters: without real game history, the search has no way to know
    that *replaying* a position ends the game as a draw, and will happily
    walk a won position into one to keep a cached "winning" eval).
    lichess-bot always resends the full move list from game start on
    every `position` command (see lib/engine_wrapper.py), so replaying
    `board`'s own move stack backwards reconstructs the true history."""
    replay = board.copy(stack=True)
    fens = []
    while replay.move_stack:
        replay.pop()
        fens.append(replay.fen())
    fens.reverse()
    return fens


def choose_move(
    evaluator: BatchedEvaluator, board: chess.Board, simulations: int, batch_size: int
) -> chess.Move | None:
    best_uci = mamba_mcts_native.search(
        board.fen(),
        evaluator,
        simulations=simulations,
        batch_size=batch_size,
        history=_game_history_fens(board),
    )
    return chess.Move.from_uci(best_uci) if best_uci else None


def warmup(evaluator: BatchedEvaluator, simulations: int, batch_size: int) -> float:
    """Run one throwaway search on the starting position before the engine
    answers its first UCI command. CUDA/ROCm kernel selection (cuDNN
    autotune, first-launch kernel compilation) is a one-time cost paid by
    whichever search hits it first -- ~80x slower than steady state,
    measured directly (see the conversation this was added from). Doing it
    here means a lichess-bot game (which spawns a fresh engine process per
    game, see lib/engine_wrapper.py's create_engine) pays that cost during
    engine startup instead of on the clock for the first real move.

    Returns the measured steady-state nodes/sec, so time management (see
    `sims_for_go`) can convert a time budget into a simulation count without
    a hardcoded, hardware-specific guess."""
    print("[bee-mamba] warming up engine...", file=sys.stderr, flush=True)
    start = time.monotonic()
    choose_move(evaluator, chess.Board(), simulations, batch_size)
    elapsed = time.monotonic() - start
    nps = simulations / elapsed if elapsed > 0 else float(simulations)
    print(f"[bee-mamba] engine warmed up ({nps:.0f} nodes/s)", file=sys.stderr, flush=True)
    return nps


def _parse_go_clock(tokens: list[str]) -> dict[str, int]:
    """Pull the clock-related fields out of a `go` command's tokens, e.g.
    `wtime 60000 btime 60000 winc 0 binc 0` or `movetime 10000`. Unknown or
    malformed tokens are ignored -- this engine doesn't support depth/nodes
    limits, so those are left for the fallback (static Simulations) path."""
    fields = {"wtime", "btime", "winc", "binc", "movetime", "movestogo"}
    parsed: dict[str, int] = {}
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] in fields:
            try:
                parsed[tokens[i]] = int(tokens[i + 1])
            except ValueError:
                pass
        i += 1
    return parsed


def _time_budget_ms(my_time_ms: int, inc_ms: int, fullmove_number: int, movestogo: int | None) -> float:
    moves_left = movestogo if movestogo else max(MIN_MOVES_LEFT, ASSUMED_TOTAL_MOVES - fullmove_number)
    budget = my_time_ms / moves_left + inc_ms * INCREMENT_WEIGHT
    budget = min(budget, my_time_ms * MAX_CLOCK_FRACTION)
    if my_time_ms < LOW_TIME_THRESHOLD_MS:
        budget = min(budget, my_time_ms * LOW_TIME_FRACTION)
    return max(MIN_BUDGET_MS, budget)


def sims_for_go(tokens: list[str], board: chess.Board, fallback_simulations: int, nps: float) -> int:
    """Decide how many simulations to spend on this move. Uses the `go`
    command's clock info when present (like any competitive UCI engine's
    own time management -- lichess-bot already forwards wtime/btime/winc/binc
    with move_overhead pre-subtracted, see lib/engine_wrapper.py's
    game_clock_time); falls back to the static Simulations option for a
    plain `go`/`go infinite` with no clock (e.g. a GUI with no time control)."""
    clock = _parse_go_clock(tokens)

    if "movetime" in clock:
        budget_ms = max(MIN_BUDGET_MS, clock["movetime"] - 50)
    else:
        time_key = "wtime" if board.turn == chess.WHITE else "btime"
        inc_key = "winc" if board.turn == chess.WHITE else "binc"
        if time_key not in clock:
            return fallback_simulations
        budget_ms = _time_budget_ms(
            clock[time_key], clock.get(inc_key, 0), board.fullmove_number, clock.get("movestogo")
        )

    sims = int((budget_ms / 1000.0) * nps)
    return max(MIN_SIMULATIONS, min(MAX_SIMULATIONS, sims))


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
    nps = warmup(evaluator, simulations, batch_size)
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
            move_sims = sims_for_go(tokens[1:], board, simulations, nps)
            print(f"[bee-mamba] {move_sims} simulations for this move", file=sys.stderr, flush=True)
            move = choose_move(evaluator, board, move_sims, batch_size)
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
