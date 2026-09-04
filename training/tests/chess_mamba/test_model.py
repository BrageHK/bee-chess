import torch
import torch.nn.functional as F

from bee_training.chess_mamba.model import N_PIECE_TYPES, N_SQUARES, ChessMamba


def test_forward_shapes():
    torch.manual_seed(0)
    model = ChessMamba(d_model=32, n_layers=2, d_state=8, n_history=7)
    B = 3
    in_dim = N_PIECE_TYPES * 8 + 8
    dummy = torch.randn(B, N_SQUARES, in_dim)

    policy_logits, value_logits = model(dummy)

    assert policy_logits.shape == (B, 64, 64)
    assert value_logits.shape == (B, 128)


def test_combined_loss_backward_touches_all_params():
    torch.manual_seed(0)
    model = ChessMamba(d_model=32, n_layers=2, d_state=8, n_history=7)
    B = 3
    in_dim = N_PIECE_TYPES * 8 + 8
    dummy = torch.randn(B, N_SQUARES, in_dim)

    policy_logits, value_logits = model(dummy)
    target_move = torch.randint(0, 64 * 64, (B,))
    target_bin = torch.randint(0, 128, (B,))
    loss = F.cross_entropy(policy_logits.reshape(B, -1), target_move) \
        + F.cross_entropy(value_logits, target_bin)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
