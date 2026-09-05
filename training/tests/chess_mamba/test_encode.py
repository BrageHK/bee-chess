import chess
import torch

from bee_training.chess_mamba.encode import IN_DIM, encode_fen, encode_position_record
from bee_training.dataset.schema import PositionRecord

STARTING_FEN = chess.STARTING_FEN


def test_starting_position_shape_and_piece_count():
    planes = encode_fen(STARTING_FEN)
    assert planes.shape == (64, IN_DIM)
    # 32 pieces on the board -> exactly 32 squares with a 1 somewhere in the piece planes
    occupied = (planes[:, :12].sum(dim=-1) > 0).sum().item()
    assert occupied == 32
    # each occupied square has exactly one piece plane set
    assert torch.equal(planes[:, :12].sum(dim=-1), (planes[:, :12].sum(dim=-1) > 0).float())


def test_starting_position_known_squares():
    planes = encode_fen(STARTING_FEN)
    # a1 (sq=0) is a white rook: piece_type=ROOK(4) -> plane index 4-1=3, white -> +0
    assert planes[0, 3] == 1.0
    # e1 (sq=4) is a white king: piece_type=KING(6) -> plane index 5, white -> +0
    assert planes[4, 5] == 1.0
    # e8 (sq=60) is a black king: plane index 5+6=11
    assert planes[60, 11] == 1.0
    # e4 (sq=28) is empty in the starting position
    assert planes[28, :12].sum() == 0


def test_starting_position_full_castling_rights():
    planes = encode_fen(STARTING_FEN)
    assert torch.all(planes[:, 12:16] == 1.0)  # K, Q, k, q all present, broadcast to every square


def test_position_with_no_castling_rights():
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    planes = encode_fen(fen)
    assert torch.all(planes[:, 12:16] == 0.0)


def test_en_passant_flag():
    fen_with_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    fen_without_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
    assert encode_fen(fen_with_ep)[0, 16] == 1.0
    assert encode_fen(fen_without_ep)[0, 16] == 0.0


def _make_record(**overrides):
    defaults = {
        "schema_version": 1, "game_id": "t", "ply": 0, "fen": STARTING_FEN, "side_to_move": "w",
        "eval_cp": 0, "eval_mate": None, "depth": 1, "best_move": "e2e4", "pv": ["e2e4"],
        "game_result": "1-0", "stockfish_version": "test",
    }
    defaults.update(overrides)
    return PositionRecord(**defaults)


def test_move_target_matches_uci_squares():
    record = _make_record(best_move="e2e4")
    _, move_target, _ = encode_position_record(record)
    move = chess.Move.from_uci("e2e4")
    assert move_target == move.from_square * 64 + move.to_square


def test_value_bin_target_monotonic_in_eval_cp():
    low = _make_record(eval_cp=-900)
    mid = _make_record(eval_cp=0)
    high = _make_record(eval_cp=900)
    _, _, bin_low = encode_position_record(low, n_value_bins=128)
    _, _, bin_mid = encode_position_record(mid, n_value_bins=128)
    _, _, bin_high = encode_position_record(high, n_value_bins=128)
    assert bin_low < bin_mid < bin_high
    assert 0 <= bin_low < 128
    assert 0 <= bin_high < 128


def test_value_bin_target_handles_mate_scores():
    mate_for_side_to_move = _make_record(eval_cp=None, eval_mate=3)
    mate_against_side_to_move = _make_record(eval_cp=None, eval_mate=-3)
    _, _, bin_pos = encode_position_record(mate_for_side_to_move, n_value_bins=128)
    _, _, bin_neg = encode_position_record(mate_against_side_to_move, n_value_bins=128)
    assert bin_pos > bin_neg
