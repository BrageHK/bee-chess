import torch

from bee_training.chess_mamba.attention_mixer import AttentionMixer


def test_shape_exact():
    torch.manual_seed(0)
    mixer = AttentionMixer(d_model=32)
    x = torch.randn(4, 64, 32)
    y = mixer(x)
    assert y.shape == (4, 64, 32)


def test_gradients_flow_through_every_parameter():
    torch.manual_seed(0)
    mixer = AttentionMixer(d_model=32)
    x = torch.randn(4, 64, 32)
    y = mixer(x)
    y.sum().backward()
    for name, p in mixer.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
