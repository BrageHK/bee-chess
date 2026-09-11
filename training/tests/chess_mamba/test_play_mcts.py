import io
from pathlib import Path

import chess
import numpy as np
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


class MaterialEvaluator:
    """A rigged `evaluate_batch` standing in for a trained model: values
    each leaf purely by material balance from the perspective of
    whichever side is to move there, with a uniform policy prior. In
    `test_choose_move_avoids_claimable_draw_when_a_winning_alternative_exists`
    below, material never changes (only kings and one knight ever move,
    no captures possible), so this is the *only* signal that could ever
    make one candidate move look better than another -- if the search
    still prefers a move that claims a draw over one that keeps this
    advantage, the native search's repetition-avoidance isn't working.
    """

    PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)  # pawn, knight, bishop, rook, queen, king
    N_VALUE_BINS = 128

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = planes.shape[0]
        policy = np.zeros((n, 4096), dtype=np.float32)  # uniform prior after softmax
        value = np.zeros((n, self.N_VALUE_BINS), dtype=np.float32)
        piece_values = np.array(self.PIECE_VALUES * 2, dtype=np.float32)  # white cols then black
        for i in range(n):
            counts = planes[i, :, :12].sum(axis=0)  # piece count per channel, this leaf
            white_material = float((counts[:6] * piece_values[:6]).sum())
            black_material = float((counts[6:12] * piece_values[6:]).sum())
            side_to_move_is_white = planes[i, 0, 18] > 0.5
            diff = white_material - black_material
            value_for_mover = diff if side_to_move_is_white else -diff
            bin_idx = self.N_VALUE_BINS - 1 if value_for_mover > 0 else 0
            value[i, bin_idx] = 20.0  # sharp spike -> softmax collapses to ~one-hot
        return policy, value


def test_choose_move_avoids_claimable_draw_when_a_winning_alternative_exists():
    # White is up a whole knight (bare kings otherwise). Starting from the
    # position below, White has already shuffled the knight a1-b3-a1-b3
    # while Black shuffled its king e8-d8-e8-d8 in lockstep, so we're back
    # to the exact starting position with White to move again -- and
    # playing a1b3 a third time would recreate the position it reaches
    # (already visited at plies 1 and 5) for a third time, a claimable
    # threefold-repetition draw. Every other legal move (any king move, or
    # the knight to c2) keeps the extra knight and is a fresh position.
    board = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
    for uci in ("a1b3", "e8d8", "b3a1", "d8e8", "a1b3", "e8d8", "b3a1", "d8e8"):
        board.push_uci(uci)

    move = choose_move(MaterialEvaluator(), board, simulations=1500, batch_size=32)

    assert move is not None
    assert move.uci() != "a1b3", (
        "must not repeat into a claimable draw while a winning alternative exists"
    )


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
