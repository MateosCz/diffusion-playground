import torch

from src.flow_matching import RFM
from src.manifolds import FlatTorus01


def test_rfm_training_target_is_shortest_geodesic_velocity():
    flow = RFM(FlatTorus01(dim=1))
    x_0 = torch.tensor([[0.9], [0.2]])
    x_data = torch.tensor([[0.1], [0.8]])
    times = torch.tensor([[0.5], [0.25]])

    returned_t, x_t, velocity = flow.sample_training_pair(
        x_data,
        x_0,
        t=times,
    )

    torch.testing.assert_close(returned_t, times)
    torch.testing.assert_close(x_t, torch.tensor([[0.0], [0.1]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        velocity,
        torch.tensor([[0.2], [-0.4]]),
        atol=1e-6,
        rtol=0,
    )


def test_rfm_canonical_model_and_sampler_are_time_first():
    flow = RFM(FlatTorus01(dim=1), integrator="rk4")
    x_0 = torch.tensor([[0.9]])

    def constant_velocity(t, x):
        assert t.shape == (x.shape[0], 1)
        return torch.full_like(x, 0.2)

    times, trajectory = flow.sample(
        constant_velocity,
        x_0,
        n_steps=4,
        return_trajectory=True,
    )

    torch.testing.assert_close(trajectory[-1], torch.tensor([[0.1]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(times, torch.linspace(0, 1, 5))
