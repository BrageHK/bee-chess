import io
from pathlib import Path

import chess
import torch

from bee_training.chess_mamba.play import choose_move, load_model, run
from bee_training.chess_mamba.train import TrainConfig, build_model


def _write_tiny_checkpoint(path: Path) -> None:
    config = TrainConfig(d_model=16, n_layers=1, n_ssm=0, d_state=4, n_value_bins=8)
    model = build_model(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.to_dict()}, path)


def test_load_model_rebuilds_architecture_from_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)

    model = load_model(checkpoint_path, "cpu")

    assert not model.training


def test_choose_move_only_ever_picks_legal_moves(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)
    model = load_model(checkpoint_path, "cpu")

    board = chess.Board()
    move = choose_move(model, board, "cpu")

    assert move in board.legal_moves


class _FakeModel(torch.nn.Module):
    """Deterministic stand-in for ChessMamba: scores a7-a8 far above
    every other square pair, so a real (untrained) model's random
    weights can't make this promotion-pruning test flaky."""

    def forward(self, planes):
        policy_logits = torch.zeros(1, 64, 64)
        policy_logits[0, chess.A7, chess.A8] = 10.0
        value_logits = torch.zeros(1, 8)
        return policy_logits, value_logits


def test_choose_move_prunes_underpromotions_to_queen():
    board = chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
    move = choose_move(_FakeModel(), board, "cpu")

    assert move.from_square == chess.A7
    assert move.to_square == chess.A8
    assert move.promotion == chess.QUEEN


def test_choose_move_returns_none_when_no_legal_moves(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)
    model = load_model(checkpoint_path, "cpu")

    # Fool's mate: white has been checkmated, no legal moves remain.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    move = choose_move(model, board, "cpu")

    assert move is None


def test_run_speaks_uci_end_to_end(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)

    in_stream = io.StringIO("uci\nisready\nposition startpos\ngo\nquit\n")
    out_stream = io.StringIO()

    run(checkpoint_path, "cpu", in_stream=in_stream, out_stream=out_stream)

    lines = out_stream.getvalue().splitlines()
    assert lines[0] == "id name Bee-Mamba"
    assert lines[2] == "uciok"
    assert lines[3] == "readyok"
    assert lines[4].startswith("bestmove ")


def test_run_applies_moves_from_position_command(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    _write_tiny_checkpoint(checkpoint_path)

    # An illegal bestmove here would mean `position ... moves ...` wasn't
    # applied to the board before `go` ran.
    in_stream = io.StringIO(
        "position startpos moves e2e4 e7e5 g1f3\ngo\nquit\n"
    )
    out_stream = io.StringIO()

    run(checkpoint_path, "cpu", in_stream=in_stream, out_stream=out_stream)

    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)

    bestmove = out_stream.getvalue().splitlines()[0].split()[1]
    assert chess.Move.from_uci(bestmove) in board.legal_moves
