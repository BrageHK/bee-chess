"""Opening position sourcing for self-play games.

Supports a real opening book (EPD, one FEN/EPD line per position, or PGN, a
file of short opening lines) as the primary diversity mechanism. Falls back
to N random legal plies when no book is given -- useful for small-scale smoke
tests that shouldn't require sourcing a book file just to validate mechanics.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.pgn


class OpeningBookError(RuntimeError):
    pass


def load_book(path: Path, pgn_plies: int = 8) -> list[str]:
    """Load a book file into a list of FEN strings.

    `pgn_plies` bounds how far into each PGN game to walk when the book is a
    PGN file of opening lines (ignored for EPD).
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith(("[Event", "[Site")):
        return _load_pgn_book(path, pgn_plies)
    return _load_epd_book(path)


def _load_epd_book(path: Path) -> list[str]:
    fens: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # EPD lines may carry opcodes after the first 4 fields; keep only
            # the position fields chess.Board can parse, defaulting halfmove/
            # fullmove counters when absent.
            fields = line.split()
            board_fields = fields[:4]
            if len(board_fields) < 4:
                continue
            fen = " ".join(board_fields) + " 0 1"
            try:
                chess.Board(fen)
            except ValueError:
                continue
            fens.append(fen)
    if not fens:
        raise OpeningBookError(f"No usable EPD positions found in {path}")
    return fens


def _load_pgn_book(path: Path, plies: int) -> list[str]:
    fens: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                if i >= plies:
                    break
                board.push(move)
            fens.append(board.fen())
    if not fens:
        raise OpeningBookError(f"No usable PGN games found in {path}")
    return fens


def sample_opening(book: list[str], rng: random.Random) -> chess.Board:
    fen = rng.choice(book)
    return chess.Board(fen)


def random_opening(rng: random.Random, plies: int) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        board.push(move)
    return board
