import io
from pathlib import Path

import chess
import torch

from bee_training.chess_mamba.play_mcts import BatchedEvaluator, choose_move, load_model, run
from bee_training.chess_mamba.train import TrainConfig, build_model

# Small enough to keep these tests fast -- correctness only, not search quality.
SIMULATIONS = 8
BATCH_SIZE = 4


def _write_tiny_checkpoint(path: Path) -> None:
    config = TrainConfig(d_model=16, n_layers=1, n_ssm=0, d_state=4, n_value_bins=8)
    model = build_model(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.to_dict()}, path)


def test_load_model_rebuilds_architecture_from_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)

    model = load_model(checkpoint_path, "cpu", "sequential")

    assert not model.training


def test_choose_move_only_ever_picks_legal_moves(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)
    model = load_model(checkpoint_path, "cpu", "sequential")
    evaluator = BatchedEvaluator(model, "cpu")

    board = chess.Board()
    move = choose_move(evaluator, board, SIMULATIONS, BATCH_SIZE)

    assert move in board.legal_moves


def test_choose_move_returns_none_when_no_legal_moves(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)
    model = load_model(checkpoint_path, "cpu", "sequential")
    evaluator = BatchedEvaluator(model, "cpu")

    # Fool's mate: white has been checkmated, no legal moves remain.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    move = choose_move(evaluator, board, SIMULATIONS, BATCH_SIZE)

    assert move is None


def test_run_speaks_uci_end_to_end(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)

    in_stream = io.StringIO("uci\nisready\nposition startpos\ngo\nquit\n")
    out_stream = io.StringIO()

    run(
        checkpoint_path,
        "cpu",
        "sequential",
        SIMULATIONS,
        BATCH_SIZE,
        in_stream=in_stream,
        out_stream=out_stream,
    )

    lines = out_stream.getvalue().splitlines()
    assert lines[0] == "id name Bee-Mamba-MCTS"
    assert lines[-3] == "uciok"
    assert lines[-2] == "readyok"
    assert lines[-1].startswith("bestmove ")


def test_run_applies_moves_from_position_command(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)

    # An illegal bestmove here would mean `position ... moves ...` wasn't
    # applied to the board before `go` ran.
    in_stream = io.StringIO("position startpos moves e2e4 e7e5 g1f3\ngo\nquit\n")
    out_stream = io.StringIO()

    run(
        checkpoint_path,
        "cpu",
        "sequential",
        SIMULATIONS,
        BATCH_SIZE,
        in_stream=in_stream,
        out_stream=out_stream,
    )

    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)

    bestmove = out_stream.getvalue().splitlines()[0].split()[1]
    assert chess.Move.from_uci(bestmove) in board.legal_moves


def test_run_setoption_changes_simulation_budget(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)

    in_stream = io.StringIO(
        f"setoption name Simulations value {SIMULATIONS}\n"
        f"setoption name BatchSize value {BATCH_SIZE}\n"
        "position startpos\ngo\nquit\n"
    )
    out_stream = io.StringIO()

    # No assertion on search quality -- just that setoption doesn't crash
    # the engine and a legal bestmove still comes out the other end.
    run(checkpoint_path, "cpu", "sequential", 800, 32, in_stream=in_stream, out_stream=out_stream)

    bestmove = out_stream.getvalue().splitlines()[0].split()[1]
    assert chess.Move.from_uci(bestmove) in chess.Board().legal_moves
