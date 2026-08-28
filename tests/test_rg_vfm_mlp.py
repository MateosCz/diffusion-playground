import pytest
import torch

from src.manifolds import FlatTorus01
from src.nn.rg_vfm_mlp import RGVFMMLP


def make_model(**kwargs) -> RGVFMMLP:
    return RGVFMMLP(
        dim=2,
        x_lifting_dim=16,
        time_embedding_half_dim=4,
        hidden_dim=[32, 32, 16],
        output_dim=2,
        manifold=FlatTorus01(dim=2),
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


@pytest.mark.parametrize("period", [1.0, 2 * torch.pi])
def test_position_encoding_is_periodic(period):
    model = make_model(
        position_fourier_bands=3,
        position_period=period,
    ).eval()
    t = torch.rand(8, 1)
    x_t = torch.rand(8, 2)

    torch.testing.assert_close(
        model(t, x_t),
        model(t, x_t + period),
        atol=1e-5,
        rtol=1e-5,
    )


def test_position_encoding_supports_a_period_per_dimension():
    period = torch.tensor([1.0, 2 * torch.pi])
    model = make_model(position_period=period).eval()
    t = torch.rand(8, 1)
    x_t = torch.rand(8, 2)

    torch.testing.assert_close(
        model(t, x_t),
        model(t, x_t + period),
        atol=1e-5,
        rtol=1e-5,
    )


def test_position_encoding_can_use_raw_coordinates():
    model = make_model(with_sincos_position=False).eval()
    captured_input = None

    def capture_lifting_input(module, args):
        del module
        nonlocal captured_input
        captured_input = args[0]

    handle = model.lifting_layer_x.register_forward_pre_hook(
        capture_lifting_input
    )
    x_t = torch.rand(8, 2)
    try:
        prediction = model(torch.rand(8, 1), x_t)
    finally:
        handle.remove()

    assert model.lifting_layer_x[0].in_features == 2
    assert prediction.shape == x_t.shape
    torch.testing.assert_close(captured_input, x_t)


def test_sincos_position_remains_the_default():
    model = make_model(position_fourier_bands=3)

    assert model.with_sincos_position is True
    assert model.lifting_layer_x[0].in_features == 2 * 2 * 3


@pytest.mark.parametrize("period", [0.0, -1.0, [1.0], [1.0, float("inf")]])
def test_position_period_must_match_dimension_and_be_positive(period):
    with pytest.raises(ValueError):
        make_model(position_period=period)


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
