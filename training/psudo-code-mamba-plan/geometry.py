"""
Precomputed chess-board geometry used by the SSM mixer.

Square indexing: sq = rank*8 + file, rank/file in 0..7 (a1=0, h1=7, a8=56, h8=63).

For each of the four "sliding piece" line families (rank, file, diagonal,
anti-diagonal) we build:
  - idx  : LongTensor (num_lines, max_len) mapping position-in-line -> square index
           (padded with 0, real validity given by `mask`)
  - mask : BoolTensor (num_lines, max_len), True where the slot is a real square

Ranks and files always have exactly 8 lines of length 8 (no padding needed).
Diagonals have 15 lines each, length 1..8 (padded to 8).

We also build a knight adjacency table: for each square, up to 8 knight-move
destination squares (padded with -1 / masked).
"""

import torch


def _pad_lines(lines, max_len):
    idx = torch.zeros(len(lines), max_len, dtype=torch.long)
    mask = torch.zeros(len(lines), max_len, dtype=torch.bool)
    for i, line in enumerate(lines):
        L = len(line)
        idx[i, :L] = torch.tensor(line, dtype=torch.long)
        mask[i, :L] = True
    return idx, mask


def build_line_families():
    """Returns a dict: name -> (idx, mask), each idx/mask shaped (num_lines, 8)."""
    ranks = [[r * 8 + f for f in range(8)] for r in range(8)]
    files = [[r * 8 + f for r in range(8)] for f in range(8)]

    main_diag = {}   # key: rank - file  (-7..7)   -- squares where this is constant
    anti_diag = {}   # key: rank + file  (0..14)
    for r in range(8):
        for f in range(8):
            sq = r * 8 + f
            main_diag.setdefault(r - f, []).append(sq)
            anti_diag.setdefault(r + f, []).append(sq)

    # order each diagonal's squares by increasing rank so "forward"/"backward"
    # scan direction is well defined and consistent across all diagonals
    main_lines = [sorted(v, key=lambda s: s // 8) for v in main_diag.values()]
    anti_lines = [sorted(v, key=lambda s: s // 8) for v in anti_diag.values()]

    families = {}
    families["rank"] = _pad_lines(ranks, 8)
    families["file"] = _pad_lines(files, 8)
    families["diag_main"] = _pad_lines(main_lines, 8)
    families["diag_anti"] = _pad_lines(anti_lines, 8)
    return families


def build_knight_adjacency():
    """Returns idx (64, 8) and mask (64, 8): knight-move destinations per square."""
    offsets = [(1, 2), (2, 1), (2, -1), (1, -2),
               (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    idx = torch.zeros(64, 8, dtype=torch.long)
    mask = torch.zeros(64, 8, dtype=torch.bool)
    for r in range(8):
        for f in range(8):
            sq = r * 8 + f
            k = 0
            for dr, df in offsets:
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    idx[sq, k] = nr * 8 + nf
                    mask[sq, k] = True
                    k += 1
    return idx, mask


if __name__ == "__main__":
    fams = build_line_families()
    for name, (idx, mask) in fams.items():
        print(name, "lines:", idx.shape[0], "max_len:", idx.shape[1],
              "lengths:", mask.sum(-1).tolist())
    kidx, kmask = build_knight_adjacency()
    print("knight adjacency:", kidx.shape, "avg degree:", kmask.sum(-1).float().mean().item())
