import pytest
import torch

from src.nn.rg_vfm_mlp import RGVFMMLP


def make_model(**kwargs) -> RGVFMMLP:
    return RGVFMMLP(
        dim=2,
        x_lifting_dim=16,
        time_embedding_half_dim=4,
        hidden_dim=[32, 32, 16],
        output_dim=2,
        **kwargs,
    )


def test_forward_matches_rg_vfm_model_interface():
    model = make_model()
    t = torch.rand(8, 1)
    x_t = torch.rand(8, 2)

    prediction = model(t, x_t)

    assert prediction.shape == x_t.shape
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_forward_accepts_flat_batch_time():
    model = make_model()

    prediction = model(torch.rand(8), torch.rand(8, 2))

    assert prediction.shape == (8, 2)


def test_position_encoding_is_periodic():
    model = make_model(position_fourier_bands=3).eval()
    t = torch.rand(8, 1)
    x_t = torch.rand(8, 2)

    torch.testing.assert_close(
        model(t, x_t),
        model(t, x_t + 2 * torch.pi),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("t", "x_t"),
    [
        (torch.rand(3, 1), torch.rand(4, 2)),
        (torch.rand(4, 2), torch.rand(4, 2)),
        (torch.rand(4, 1), torch.rand(4, 3)),
    ],
)
def test_forward_rejects_invalid_shapes(t, x_t):
    with pytest.raises(ValueError):
        make_model()(t, x_t)
