import copy

import pytest
import torch
import torch.nn.functional as F

from bee_training.chess_mamba.mamba_core import MambaBlock, SelectiveSSM, get_scan_fn
from bee_training.chess_mamba.triton_scan import TRITON_AVAILABLE


def test_gradients_flow_through_every_parameter():
    torch.manual_seed(0)
    blk = MambaBlock(d_model=32)
    x = torch.randn(4, 8, 32)
    y = blk(x)
    y.sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"


def test_padding_is_an_exact_no_op():
    """Garbage in the masked region of a MambaBlock input must not change
    the real-region output at all -- this is what lets rank/file lines
    (length 8) and diagonals (length 1-8) share the same scan code."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    mask = torch.tensor([True, True, True, False, False, False, False, False])

    x2 = x.clone()
    x2[:, 3:] = torch.randn(4, 5, 32) * 100  # garbage in the padded region

    torch.manual_seed(1)
    m = MambaBlock(d_model=32)
    y_a = m(x, mask=mask)
    y_b = m(x2, mask=mask)

    assert torch.equal(y_a[:, :3], y_b[:, :3])


def _sequential_reference_scan(ssm: SelectiveSSM, x, mask=None):
    """Reference selective-scan using a plain Python for-loop instead of
    pscan -- computes the exact same recurrence a different way, used only
    to check the pscan backend didn't silently change the numerics."""
    B, L, D = x.shape
    N = ssm.d_state
    A = -torch.exp(ssm.A_log.float())

    x_dbl = ssm.x_proj(x)
    dt_rank = x_dbl.shape[-1] - 2 * N
    dt, Bmat, Cmat = torch.split(x_dbl, [dt_rank, N, N], dim=-1)
    dt = F.softplus(ssm.dt_proj(dt))

    if mask is not None:
        m = mask.unsqueeze(0).expand(B, L) if mask.dim() == 1 else mask
        dt = dt * m.unsqueeze(-1).to(dt.dtype)

    Abar = torch.exp(dt.unsqueeze(-1) * A)
    Bbar = dt.unsqueeze(-1) * Bmat.unsqueeze(2)
    BX = Bbar * x.unsqueeze(-1)

    h = x.new_zeros(B, D, N)
    hs = []
    for t in range(L):
        h = Abar[:, t] * h + BX[:, t]
        hs.append(h)
    hs = torch.stack(hs, dim=1)

    y = torch.einsum("bldn,bln->bld", hs, Cmat)
    y = y + x * ssm.D
    if mask is not None:
        m = mask.unsqueeze(0).expand(B, L) if mask.dim() == 1 else mask
        y = y * m.unsqueeze(-1).to(y.dtype)
    return y


def test_pscan_backend_matches_sequential_reference_unpadded():
    torch.manual_seed(0)
    ssm = SelectiveSSM(d_inner=16, d_state=8)
    x = torch.randn(3, 8, 16)

    y_pscan = ssm(x)
    y_seq = _sequential_reference_scan(ssm, x)

    assert torch.allclose(y_pscan, y_seq, atol=1e-5)


def test_pscan_backend_matches_sequential_reference_padded():
    torch.manual_seed(0)
    ssm = SelectiveSSM(d_inner=16, d_state=8)
    x = torch.randn(3, 8, 16)
    mask = torch.tensor([True, True, True, True, True, False, False, False])

    y_pscan = ssm(x, mask=mask)
    y_seq = _sequential_reference_scan(ssm, x, mask=mask)

    assert torch.allclose(y_pscan, y_seq, atol=1e-5)


@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="triton scan_backend requires Triton + a CUDA/ROCm GPU",
)
def test_triton_backend_matches_pscan_backend():
    """scan_backend='triton' must produce the same result as the default
    'pscan' backend given identical weights -- the choice of scan
    implementation is purely a performance knob, never a numerics one."""
    torch.manual_seed(0)
    blk_pscan = MambaBlock(d_model=32, d_state=8, scan_backend="pscan").cuda()
    blk_triton = copy.deepcopy(blk_pscan)
    blk_triton.ssm._scan = get_scan_fn("triton")

    x = torch.randn(3, 8, 32, device="cuda")
    mask = torch.tensor([True, True, True, True, True, False, False, False], device="cuda")

    y_pscan = blk_pscan(x, mask=mask)
    y_triton = blk_triton(x, mask=mask)

    assert torch.allclose(y_pscan, y_triton, atol=1e-4)
