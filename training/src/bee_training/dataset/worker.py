"""Per-process self-play loop: open one Stockfish engine, play an assigned
range of games, write shards, and update the shared manifest.
"""

from __future__ import annotations

import random
import signal
from contextlib import AbstractContextManager
from pathlib import Path

import chess.engine

from bee_training.dataset import manifest as manifest_mod
from bee_training.dataset.config import SelfPlayConfig
from bee_training.dataset.openings import load_book
from bee_training.dataset.schema import append_jsonl
from bee_training.dataset.selfplay import play_one_game


def _shard_paths(config: SelfPlayConfig, worker_id: str) -> tuple[Path, Path, Path]:
    shard_dir = config.run_dir() / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"worker-{worker_id}"
    return (
        shard_dir / f"{prefix}.positions.jsonl",
        shard_dir / f"{prefix}.games.jsonl",
        shard_dir / f"{prefix}.pgn",
    )


def run_worker(
    worker_id: str,
    config: SelfPlayConfig,
    stockfish_path: Path,
    lock: AbstractContextManager | None,
) -> None:
    # generate.py's orchestrator sends SIGTERM (not just SIGINT/Ctrl+C) to stop
    # a run. Without a handler, SIGTERM kills this process immediately without
    # running the `with engine:` block's __exit__, orphaning the Stockfish
    # subprocess. Raising here instead unwinds through that block, so `engine`
    # gets a clean `quit()`.
    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    state = manifest_mod.load(config).workers[worker_id]
    remaining = state.remaining()
    if not remaining:
        return

    book = load_book(Path(config.opening_book)) if config.opening_book else None
    rng = random.Random(config.seed + int(worker_id))

    positions_path, games_path, pgn_path = _shard_paths(config, worker_id)

    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 32})
        for game_index in remaining:
            game_id = f"{config.run_id}-{game_index}"
            outcome = play_one_game(engine, config, game_id, rng, book)

            append_jsonl(positions_path, [p.to_json() for p in outcome.positions])
            append_jsonl(games_path, [outcome.game.to_json()])
            if config.emit_pgn:
                with pgn_path.open("a", encoding="utf-8") as f:
                    f.write(outcome.pgn_text)
                    f.write("\n\n")

            manifest_mod.mark_completed(config, worker_id, game_index, lock=lock)
