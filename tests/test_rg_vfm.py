import torch

from src.flow_matching.rg_vfm import RGVFM
from src.manifolds.flat_torus import FlatTorus01


def make_flow(**kwargs):
    return RGVFM(
        FlatTorus01(dim=1),
        max_velocity_scale=None,
        **kwargs,
    )


def test_forward_sample_returns_endpoint_target_and_time():
    flow = make_flow()
    x0 = torch.tensor([[0.9], [0.2]])
    x1 = torch.tensor([[0.1], [0.8]])
    ts = torch.tensor([[0.5], [0.25]])

    x_t, target, returned_t = flow.sample_forward(
        x1,
        x0=x0,
        ts=ts,
        return_time=True,
    )

    torch.testing.assert_close(x_t, torch.tensor([[0.0], [0.1]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(target, x1)
    torch.testing.assert_close(returned_t, ts)


def test_flow_matching_training_api_is_time_first():
    flow = make_flow()
    x0 = torch.tensor([[0.9], [0.2]])
    x1 = torch.tensor([[0.1], [0.8]])
    ts = torch.tensor([[0.5], [0.25]])

    returned_t, x_t, target = flow.sample_training_pair(x1, x0, t=ts)

    torch.testing.assert_close(returned_t, ts)
    torch.testing.assert_close(x_t, torch.tensor([[0.0], [0.1]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(target, x1)


def test_forward_can_return_shortest_geodesic_trajectory():
    flow = make_flow()
    x0 = torch.tensor([[0.9]])
    x1 = torch.tensor([[0.1]])

    trajectory, targets, times = flow.sample_forward(
        x1,
        x0=x0,
        sample_trajectory=True,
        n_steps=4,
        return_time=True,
    )

    expected = torch.tensor([[[0.9]], [[0.95]], [[0.0]], [[0.05]], [[0.1]]])
    torch.testing.assert_close(trajectory, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(targets, x1.unsqueeze(0).expand_as(trajectory))
    torch.testing.assert_close(times, torch.linspace(0, 1, 5))


def test_loss_uses_geodesic_distance_across_periodic_boundary():
    flow = make_flow()
    pred = torch.tensor([[0.95]])
    target = torch.tensor([[0.05]])

    torch.testing.assert_close(flow.loss(pred, target), torch.tensor(0.01))


def test_backward_step_converts_endpoint_to_velocity():
    flow = make_flow()
    x_t = torch.tensor([[0.9]])
    endpoint = torch.tensor([[0.1]])

    x_next = flow.sample_backward_step(x_t, t=0.0, pred=endpoint, dt=0.25)

    torch.testing.assert_close(x_next, torch.tensor([[0.95]]), atol=1e-6, rtol=0)


def test_backward_sampling_reaches_constant_predicted_endpoint_and_tracks_path():
    flow = make_flow()
    x0 = torch.tensor([[0.9], [0.2]])
    endpoint = torch.tensor([[0.1], [0.8]])

    def endpoint_model(x_t, t):
        assert t.shape == (x_t.shape[0], 1)
        return endpoint

    trajectory, times = flow.sample_backward(
        x0,
        endpoint_model,
        n_steps=4,
        sample_trajectory=True,
        return_time=True,
    )

    torch.testing.assert_close(trajectory[-1], endpoint, atol=1e-6, rtol=0)
    assert trajectory.shape == (5, 2, 1)
    torch.testing.assert_close(times, torch.linspace(0, 1, 5))


def test_canonical_sample_uses_time_first_model():
    flow = make_flow()
    x0 = torch.tensor([[0.9], [0.2]])
    endpoint = torch.tensor([[0.1], [0.8]])

    def endpoint_model(t, x_t):
        assert t.shape == (x_t.shape[0], 1)
        return endpoint

    times, trajectory = flow.sample(
        endpoint_model,
        x0,
        n_steps=4,
        return_trajectory=True,
    )

    torch.testing.assert_close(trajectory[-1], endpoint, atol=1e-6, rtol=0)
    torch.testing.assert_close(times, torch.linspace(0, 1, 5))


def test_stabilized_endpoint_field_supports_higher_order_integrators():
    x0 = torch.tensor([[0.2]])

    for method in ("heun", "rk4"):
        flow = RGVFM(FlatTorus01(dim=1), integrator=method)
        times, trajectory = flow.sample(
            lambda t, x: torch.full_like(x, 0.4),
            x0,
            n_steps=4,
            return_trajectory=True,
        )
        assert times.shape == (5,)
        assert trajectory.shape == (5, 1, 1)
        assert torch.isfinite(trajectory).all()
