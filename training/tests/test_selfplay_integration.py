"""End-to-end self-play test against a real Stockfish binary.

Downloads/verifies the real Stockfish release and drives real search, so it's
skipped by default (CI's default `uv run pytest` must stay fast/offline).
Opt in explicitly:

    RUN_STOCKFISH_INTEGRATION_TESTS=1 uv run pytest tests/test_selfplay_integration.py
"""

from __future__ import annotations

import io
import os
import random

import chess
import chess.engine
import chess.pgn
import pytest

from bee_training.dataset.config import SelfPlayConfig
from bee_training.dataset.selfplay import play_one_game
from bee_training.dataset.stockfish_fetch import ensure_stockfish

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_STOCKFISH_INTEGRATION_TESTS") != "1",
    reason="Set RUN_STOCKFISH_INTEGRATION_TESTS=1 to run (downloads and runs a real Stockfish binary).",
)


@pytest.fixture(scope="module")
def stockfish_path():
    path, _tag = ensure_stockfish(version="latest")
    return path


def _config(tmp_path, **overrides) -> SelfPlayConfig:
    defaults = {
        "games": 1,
        "workers": 1,
        "limit_kind": "nodes",
        "limit_value": 2000,
        "opening_book": None,
        "opening_plies": 6,
        "resign_cp": 1000,
        "resign_plies": 6,
        "draw_cp": 10,
        "draw_plies": 6,
        "draw_min_ply": 20,
        "max_plies": 60,
        "stockfish_version": "latest",
        "output_dir": str(tmp_path),
        "run_id": "integration-test",
        "seed": 0,
    }
    defaults.update(overrides)
    return SelfPlayConfig(**defaults)


def test_play_one_game_produces_valid_positions_and_pgn(tmp_path, stockfish_path) -> None:
    config = _config(tmp_path)
    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        outcome = play_one_game(engine, config, "game-0", random.Random(0), book=None)

    assert outcome.positions
    for position in outcome.positions:
        chess.Board(position.fen)  # must not raise
        chess.Move.from_uci(position.best_move)
        assert position.game_result == outcome.game.result

    game = chess.pgn.read_game(io.StringIO(outcome.pgn_text))
    assert game is not None
    assert game.headers["Result"] == outcome.game.result


def test_replaying_recorded_moves_ends_in_recorded_result(tmp_path, stockfish_path) -> None:
    config = _config(tmp_path)
    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        outcome = play_one_game(engine, config, "game-1", random.Random(1), book=None)

    board = chess.Board(outcome.positions[0].fen)
    for position in outcome.positions:
        move = chess.Move.from_uci(position.best_move)
        assert move in board.legal_moves
        board.push(move)

    if outcome.game.termination == "checkmate":
        assert board.is_checkmate()
