import pytest
import torch
import torch.nn.functional as F

from bee_training.chess_mamba.triton_scan import TRITON_AVAILABLE, triton_pscan

pytestmark = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="triton_scan requires Triton + a CUDA/ROCm GPU",
)


def _sequential_reference_scan(Abar, BX):
    """h_t = Abar_t * h_{t-1} + BX_t, h_{-1}=0 -- same recurrence as pscan."""
    B, L, D, N = Abar.shape
    h = Abar.new_zeros(B, D, N)
    hs = []
    for t in range(L):
        h = Abar[:, t] * h + BX[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


def _make_inputs(B, L, D, N, mask=None, seed=0, requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = F.softplus(torch.randn(B, L, D, generator=g)).cuda()
    A = -torch.exp(torch.randn(D, N, generator=g)).cuda()
    Bmat = torch.randn(B, L, N, generator=g).cuda()
    x = torch.randn(B, L, D, generator=g).cuda()

    if mask is not None:
        m = mask.cuda()
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(B, L)
        dt = dt * m.unsqueeze(-1).to(dt.dtype)

    Abar = torch.exp(dt.unsqueeze(-1) * A)
    BX = (dt.unsqueeze(-1) * Bmat.unsqueeze(2)) * x.unsqueeze(-1)

    Abar = Abar.contiguous().requires_grad_(requires_grad)
    BX = BX.contiguous().requires_grad_(requires_grad)
    return Abar, BX


def test_forward_matches_sequential_reference_unpadded():
    Abar, BX = _make_inputs(B=3, L=8, D=16, N=8, seed=0)
    assert torch.allclose(triton_pscan(Abar, BX), _sequential_reference_scan(Abar, BX), atol=1e-4)


def test_forward_matches_sequential_reference_padded():
    mask = torch.tensor([True, True, True, True, True, False, False, False])
    Abar, BX = _make_inputs(B=3, L=8, D=16, N=8, mask=mask, seed=1)
    y_triton = triton_pscan(Abar, BX)
    y_ref = _sequential_reference_scan(Abar, BX)
    assert torch.allclose(y_triton, y_ref, atol=1e-4)
    # explicit padding no-op check, same spirit as test_mamba_core.py
    assert torch.allclose(y_triton[:, :5], y_ref[:, :5], atol=1e-4)


def test_forward_matches_mambapy_pscan():
    from mambapy.pscan import pscan as mambapy_pscan

    Abar, BX = _make_inputs(B=4, L=8, D=32, N=16, seed=2)
    assert torch.allclose(triton_pscan(Abar, BX), mambapy_pscan(Abar, BX), atol=1e-4)


@pytest.mark.parametrize("mask", [None, torch.tensor([True] * 5 + [False] * 3)])
def test_gradients_match_sequential_reference(mask):
    Abar_t, BX_t = _make_inputs(B=3, L=8, D=16, N=8, mask=mask, seed=0, requires_grad=True)
    Abar_r = Abar_t.detach().clone().requires_grad_(True)
    BX_r = BX_t.detach().clone().requires_grad_(True)

    grad_out = torch.randn(3, 8, 16, 8, device="cuda")
    triton_pscan(Abar_t, BX_t).backward(grad_out)
    _sequential_reference_scan(Abar_r, BX_r).backward(grad_out)

    assert torch.allclose(Abar_t.grad, Abar_r.grad, atol=1e-3)
    assert torch.allclose(BX_t.grad, BX_r.grad, atol=1e-3)
