import random

import chess
import pytest

from bee_training.dataset.openings import (
    OpeningBookError,
    load_book,
    random_opening,
    sample_opening,
)


def test_random_opening_produces_legal_board() -> None:
    board = random_opening(random.Random(0), plies=8)
    assert board.is_valid()
    assert board.ply() >= 0


def test_random_opening_is_deterministic_given_a_seed() -> None:
    board_a = random_opening(random.Random(42), plies=8)
    board_b = random_opening(random.Random(42), plies=8)
    assert board_a.fen() == board_b.fen()


def test_random_opening_varies_across_seeds() -> None:
    fens = {random_opening(random.Random(seed), plies=8).fen() for seed in range(20)}
    assert len(fens) > 1


def test_load_epd_book(tmp_path) -> None:
    book_path = tmp_path / "book.epd"
    book_path.write_text(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -\n"
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -\n",
        encoding="utf-8",
    )
    book = load_book(book_path)
    assert len(book) == 2
    for fen in book:
        chess.Board(fen)  # must parse without raising


def test_load_epd_book_skips_blank_and_comment_lines(tmp_path) -> None:
    book_path = tmp_path / "book.epd"
    book_path.write_text(
        "# a comment\n\nrnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -\n",
        encoding="utf-8",
    )
    book = load_book(book_path)
    assert len(book) == 1


def test_load_epd_book_empty_raises(tmp_path) -> None:
    book_path = tmp_path / "book.epd"
    book_path.write_text("# nothing usable here\n", encoding="utf-8")
    with pytest.raises(OpeningBookError):
        load_book(book_path)


def test_load_pgn_book(tmp_path) -> None:
    book_path = tmp_path / "book.pgn"
    book_path.write_text(
        '[Event "Test"]\n[Site "?"]\n[Date "????.??.??"]\n[Round "?"]\n'
        '[White "?"]\n[Black "?"]\n[Result "*"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 *\n",
        encoding="utf-8",
    )
    book = load_book(book_path, pgn_plies=2)
    assert len(book) == 1
    board = chess.Board(book[0])
    assert board.is_valid()
    assert board.ply() == 2  # walked 2 of the game's 4 plies, per pgn_plies=2
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)


def test_sample_opening_draws_from_book() -> None:
    book = [chess.Board().fen()]
    board = sample_opening(book, random.Random(0))
    assert board.fen() == chess.Board().fen()
