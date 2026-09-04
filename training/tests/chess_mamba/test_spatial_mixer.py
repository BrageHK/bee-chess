import torch

from bee_training.chess_mamba.geometry import build_knight_adjacency, build_line_families
from bee_training.chess_mamba.spatial_mixer import SpatialMixer


def test_shape_exact():
    torch.manual_seed(0)
    mixer = SpatialMixer(d_model=32)
    x = torch.randn(4, 64, 32)
    y = mixer(x)
    assert y.shape == (4, 64, 32)


def test_gradients_flow_through_every_parameter():
    torch.manual_seed(0)
    mixer = SpatialMixer(d_model=32)
    x = torch.randn(4, 64, 32)
    y = mixer(x)
    y.sum().backward()
    for name, p in mixer.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"


def _related_squares(sq: int) -> set[int]:
    related = {sq}
    for idx, mask in build_line_families().values():
        for line_idx, line_mask in zip(idx, mask):
            real = line_idx[line_mask].tolist()
            if sq in real:
                related.update(real)
    kidx, kmask = build_knight_adjacency()
    related.update(kidx[sq][kmask[sq]].tolist())
    return related


def test_changing_one_square_only_affects_related_squares():
    """Swapping the piece on a single square should change that square's
    own output and the outputs of squares on its rank/file/diagonals/knight
    graph, but not unrelated squares -- catches scatter-index bugs that
    shape/gradient tests alone can't."""
    torch.manual_seed(0)
    mixer = SpatialMixer(d_model=16)
    mixer.eval()

    x = torch.randn(1, 64, 16)
    sq = 27  # d4, an interior square with rank/file/both diagonals full length

    with torch.no_grad():
        y_before = mixer(x)

        x2 = x.clone()
        x2[:, sq, :] = torch.randn(16)
        y_after = mixer(x2)

    diff = (y_after - y_before).abs().sum(dim=-1).squeeze(0)  # (64,)
    changed = set(torch.nonzero(diff > 1e-6).flatten().tolist())
    related = _related_squares(sq)
    unrelated = set(range(64)) - related

    assert sq in changed
    assert changed.issubset(related), f"changed squares outside expected set: {changed - related}"
    assert changed & unrelated == set()
