"""
Minimal board encoder: FEN -> (64, in_dim) plane tensor for `ChessMamba`,
plus target extraction from this project's self-play `PositionRecord`
schema (`bee_training.dataset.schema`).

This is a deliberately minimal slice of the full encoder described in
CHESSMAMBA_PLAN.md Section 2 -- single position only (no game-history
planes, no side-to-move board flip), matching `ChessMamba`'s current
fixed `in_dim = 12*(n_history+1) + 8` budget with `n_history=0`. It
exists to let benchmarks (and later, real training) run actual self-play
data through the model instead of random tensors -- it is NOT the
validated, training-ready encoder Phase 4 calls for; that needs the
history planes, the side-to-move flip, and a real correctness test
against hand-picked FENs before anything is trained on it for real.

Per-square layout, `in_dim = 20`:
  [0:12]  one-hot piece-per-square (6 piece types x 2 colors); empty
          squares are all-zero.
  [12:20] 8 auxiliary scalars, broadcast identically to every square:
          castling rights (K, Q, k, q), en-passant-available flag,
          halfmove clock / 100, side-to-move flag, one reserved/unused
          slot (kept only to match ChessMamba's fixed "+8" budget).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chess
import torch

from bee_training.dataset.schema import PositionRecord, read_jsonl

N_PIECE_TYPES = 12
N_AUX = 8
IN_DIM = N_PIECE_TYPES + N_AUX


def encode_fen(fen: str) -> torch.Tensor:
    """FEN -> (64, IN_DIM) float tensor. Square index matches this
    project's `geometry.py` convention (sq = rank*8 + file, a1=0..h8=63)
    -- which is also `python-chess`'s own `chess.SQUARES` numbering, so
    no reindexing is needed."""
    board = chess.Board(fen)
    planes = torch.zeros(64, IN_DIM)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            planes[sq, plane] = 1.0

    aux = torch.zeros(N_AUX)
    aux[0] = float(board.has_kingside_castling_rights(chess.WHITE))
    aux[1] = float(board.has_queenside_castling_rights(chess.WHITE))
    aux[2] = float(board.has_kingside_castling_rights(chess.BLACK))
    aux[3] = float(board.has_queenside_castling_rights(chess.BLACK))
    aux[4] = float(board.ep_square is not None)
    aux[5] = board.halfmove_clock / 100.0
    aux[6] = float(board.turn == chess.WHITE)
    # aux[7] reserved, left 0.0
    planes[:, N_PIECE_TYPES:] = aux
    return planes


def encode_position_record(record: PositionRecord, n_value_bins: int = 128,
                            cp_clip: float = 1000.0) -> tuple[torch.Tensor, int, int]:
    """Returns (board_planes (64, IN_DIM), move_target, value_bin_target).

    move_target: from_square*64 + to_square, matching FromToPolicyHead's
    flattened (64*64) logits.
    value_bin_target: eval_cp (or a saturating stand-in for eval_mate)
    linearly binned into `n_value_bins` buckets over [-cp_clip, cp_clip].
    This is a simple linear binning for benchmark purposes -- real
    training should use the HL-Gauss transform CHESSMAMBA_PLAN.md
    Section 6 calls for, not this.
    """
    planes = encode_fen(record.fen)

    move = chess.Move.from_uci(record.best_move)
    move_target = move.from_square * 64 + move.to_square

    if record.eval_cp is not None:
        cp = record.eval_cp
    else:
        cp = cp_clip if record.eval_mate > 0 else -cp_clip
    cp = max(-cp_clip, min(cp_clip, cp))
    value_bin_target = int((cp + cp_clip) / (2 * cp_clip) * n_value_bins)
    value_bin_target = max(0, min(n_value_bins - 1, value_bin_target))

    return planes, move_target, value_bin_target


def _iter_jsonl_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Like `read_jsonl`, but yields one line at a time instead of reading
    the whole file into a list first -- `read_jsonl` itself can't do this
    (it's relied on elsewhere as an eager, indexable list), and eagerly
    materializing a hundreds-of-MB shard file just to honor a
    `max_records` cap a few thousand lines in defeats the point of the
    cap."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_all_records(shard_paths, max_records: int | None = None) -> list[PositionRecord]:
    """Reads records from `shard_paths` (list of `*.positions.jsonl` files)
    into memory. At this project's original data scale (tens of thousands
    of positions) loading everything was fine; a shard file is now one
    self-play worker's whole *still-growing* output (hundreds of MB
    each), and a `PositionRecord` instance costs several times its raw
    JSON-line size once parsed (dataclass overhead, a separately-boxed
    `pv` string per ply, etc.) -- loading every shard in full can run a
    host out of memory. Pass `max_records` to stop early (mid-file, if
    need be) once that many are loaded; each shard is one worker's
    continuous self-play stream, so a prefix of it is still an unbiased
    sample of the same distribution, just less data."""
    records: list[PositionRecord] = []
    for path in shard_paths:
        for d in _iter_jsonl_lines(Path(path)):
            records.append(PositionRecord.from_dict(d))
            if max_records is not None and len(records) >= max_records:
                return records
    return records


class PositionDataset(torch.utils.data.Dataset):
    """Wraps a list of `PositionRecord`s, encoding each one lazily in
    `__getitem__` (not all up front) so a `DataLoader` with
    `num_workers > 0` can parallelize the python-chess FEN parsing across
    processes instead of it being a serial bottleneck before training
    even starts."""

    def __init__(self, records: list[PositionRecord], n_value_bins: int = 128):
        self.records = records
        self.n_value_bins = n_value_bins

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return encode_position_record(self.records[idx], n_value_bins=self.n_value_bins)


def load_real_batch(shard_paths, batch_size: int, n_value_bins: int = 128,
                     device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reads real self-play positions from `shard_paths` (list of
    `*.positions.jsonl` files) and encodes the first `batch_size` of them
    into (board_planes, move_targets, value_bin_targets) tensors."""
    records: list[PositionRecord] = []
    for path in shard_paths:
        for d in read_jsonl(Path(path)):
            records.append(PositionRecord.from_dict(d))
            if len(records) >= batch_size:
                break
        if len(records) >= batch_size:
            break
    if len(records) < batch_size:
        raise ValueError(f"only found {len(records)} real positions, need {batch_size}")

    planes_list, move_targets, value_targets = [], [], []
    for record in records[:batch_size]:
        planes, move_t, value_t = encode_position_record(record, n_value_bins=n_value_bins)
        planes_list.append(planes)
        move_targets.append(move_t)
        value_targets.append(value_t)

    x = torch.stack(planes_list).to(device)
    target_move = torch.tensor(move_targets, dtype=torch.long, device=device)
    target_bin = torch.tensor(value_targets, dtype=torch.long, device=device)
    return x, target_move, target_bin
