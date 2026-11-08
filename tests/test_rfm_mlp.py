import torch

from src.manifolds import FlatTorus01
from src.nn.rfm_mlp import RFMMLP


def make_model(**kwargs) -> RFMMLP:
    return RFMMLP(
        dim=2,
        x_lifting_dim=16,
        time_embedding_half_dim=4,
        hidden_dim=[32, 16],
        output_dim=2,
        position_fourier_bands=4,
        position_period=1.0,
        manifold=FlatTorus01(dim=2),
        **kwargs,
    )


def test_rfm_mlp_has_periodic_input_features():
    model = make_model().eval()
    t = torch.rand(8, 1)
    x_t = torch.rand(8, 2)

    torch.testing.assert_close(
        model(t, x_t),
        model(t, x_t + 1.0),
        atol=1e-5,
        rtol=1e-5,
    )


def test_rfm_mlp_does_not_wrap_tangent_output():
    model = make_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_layer.bias.copy_(torch.tensor([1.5, -0.75]))

    velocity = model(torch.rand(4, 1), torch.rand(4, 2))

    torch.testing.assert_close(
        velocity,
        torch.tensor([[1.5, -0.75]]).expand(4, -1),
    )


def test_rfm_mlp_output_backpropagates():
    model = make_model()

    prediction = model(torch.rand(8, 1), torch.rand(8, 2))
    prediction.square().mean().backward()

    assert prediction.shape == (8, 2)
    assert all(parameter.grad is not None for parameter in model.parameters())
