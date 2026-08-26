import torch

from src.flow_matching import RGVFM
from src.manifolds import FlatTorus01


def test_flat_torus_declares_and_round_trips_ambient_embedding():
    torus_1d = FlatTorus01(dim=1)
    torus_2d = FlatTorus01(dim=2)
    assert torus_1d.ambient_dim == 2
    assert torus_2d.ambient_dim == 4

    points = torch.tensor([[0.0, 0.25], [0.5, 0.75]], dtype=torch.float64)
    ambient = torus_2d.to_ambient(points)

    assert ambient.shape == (2, 4)
    torch.testing.assert_close(
        torus_2d.from_ambient(ambient),
        points,
        atol=1e-7,
        rtol=0,
    )


def test_flat_torus_projects_each_ambient_pair_to_unit_circle():
    manifold = FlatTorus01(dim=2)
    ambient = torch.tensor([[3.0, 4.0, 0.0, 0.0]])

    projected = manifold.project_ambient(ambient).reshape(1, 2, 2)
    norms = torch.linalg.vector_norm(projected, dim=-1)

    torch.testing.assert_close(norms, torch.ones_like(norms))
    torch.testing.assert_close(projected[0, 1], torch.tensor([1.0, 0.0]))


def test_extrinsic_prior_uses_manifold_ambient_dimension():
    data_1d = torch.zeros(7, 1, dtype=torch.float64)
    data_2d = torch.zeros(7, 2, dtype=torch.float64)
    flow_1d = RGVFM(FlatTorus01(dim=1), support="extrinsic")
    flow_2d = RGVFM(FlatTorus01(dim=2), support="extrinsic")

    prior_1d = flow_1d.sample_prior(data_1d)
    prior_2d = flow_2d.sample_prior(data_2d)

    assert prior_1d.shape == (7, 2)
    assert prior_2d.shape == (7, 4)
    assert prior_1d.dtype == data_1d.dtype
    assert prior_2d.dtype == data_2d.dtype
    assert flow_1d.model_dim == 2
    assert flow_2d.model_dim == 4


def test_extrinsic_training_uses_linear_ambient_path_and_endpoint_target():
    manifold = FlatTorus01(dim=1)
    flow = RGVFM(manifold, support="extrinsic")
    x_data = torch.tensor([[0.0], [0.25]])
    x_base = torch.zeros(2, 2)
    times = torch.full((2, 1), 0.5)
    endpoint = manifold.to_ambient(x_data)

    returned_t, x_t, target = flow.sample_training_pair(
        x_data,
        x_base,
        t=times,
    )

    torch.testing.assert_close(returned_t, times)
    torch.testing.assert_close(x_t, 0.5 * endpoint, atol=1e-6, rtol=0)
    torch.testing.assert_close(target, endpoint, atol=1e-6, rtol=0)


def test_extrinsic_loss_is_torus_geodesic_distance_not_ambient_mse():
    manifold = FlatTorus01(dim=1)
    flow = RGVFM(manifold, support="extrinsic")
    prediction = 2 * manifold.to_ambient(torch.tensor([[0.95]]))
    target = manifold.to_ambient(torch.tensor([[0.05]]))

    torch.testing.assert_close(
        flow.loss(prediction, target),
        torch.tensor(0.01),
        atol=1e-6,
        rtol=0,
    )


def test_extrinsic_geodesic_loss_is_differentiable():
    manifold = FlatTorus01(dim=1)
    flow = RGVFM(manifold, support="extrinsic")
    prediction = (
        2 * manifold.to_ambient(torch.tensor([[0.95]]))
    ).requires_grad_(True)
    target = manifold.to_ambient(torch.tensor([[0.05]]))

    loss = flow.loss(prediction, target)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_extrinsic_sampling_stays_in_ambient_space_and_decodes_to_torus():
    manifold = FlatTorus01(dim=1)
    flow = RGVFM(
        manifold,
        support="extrinsic",
        max_velocity_scale=None,
    )
    x_base = torch.tensor([[0.0, 0.0], [0.5, -0.5]])
    endpoint = manifold.to_ambient(torch.tensor([[0.25], [0.75]]))

    samples = flow.sample(lambda t, x: endpoint, x_base, n_steps=4)
    decoded = flow.to_intrinsic(samples)

    assert samples.shape == (2, 2)
    torch.testing.assert_close(samples, endpoint, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        decoded,
        torch.tensor([[0.25], [0.75]]),
        atol=1e-6,
        rtol=0,
    )
