import pytest

from bee_training.chess_mamba.geometry import build_knight_adjacency, build_line_families


def test_rank_file_lines_have_no_padding():
    families = build_line_families()
    for name in ("rank", "file"):
        idx, mask = families[name]
        assert idx.shape == (8, 8)
        assert mask.shape == (8, 8)
        assert mask.all(), f"{name} lines should never need padding"


def test_diagonal_lengths_match_known_pattern():
    families = build_line_families()

    idx_main, mask_main = families["diag_main"]
    idx_anti, mask_anti = families["diag_anti"]

    assert idx_main.shape == (15, 8)
    assert idx_anti.shape == (15, 8)

    main_lengths = mask_main.sum(-1).tolist()
    anti_lengths = mask_anti.sum(-1).tolist()

    assert main_lengths == [8, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1]
    assert anti_lengths == [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]


def test_diagonal_squares_ordered_by_increasing_rank():
    families = build_line_families()
    for name in ("diag_main", "diag_anti"):
        idx, mask = families[name]
        for line_idx, line_mask in zip(idx, mask):
            real = line_idx[line_mask]
            ranks = (real // 8).tolist()
            assert ranks == sorted(ranks), f"{name} line not ordered by increasing rank"


def test_knight_adjacency_average_out_degree():
    idx, mask = build_knight_adjacency()
    assert idx.shape == (64, 8)
    assert mask.shape == (64, 8)
    avg_degree = mask.sum(-1).float().mean().item()
    assert avg_degree == pytest.approx(5.25)
