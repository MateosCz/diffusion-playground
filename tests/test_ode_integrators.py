import math

import torch

from src.ode import EulerIntegrator, HeunIntegrator, RK4Integrator


def _exponential_field(t, x):
    del t
    return x


def test_integrators_solve_exponential_with_expected_accuracy_order():
    x0 = torch.ones(1, 1, dtype=torch.float64)
    times = torch.linspace(0, 1, 5, dtype=torch.float64)
    exact = math.e

    euler = EulerIntegrator().integrate(_exponential_field, x0, times).item()
    heun = HeunIntegrator().integrate(_exponential_field, x0, times).item()
    rk4 = RK4Integrator().integrate(_exponential_field, x0, times).item()

    assert abs(rk4 - exact) < abs(heun - exact) < abs(euler - exact)


def test_integrator_can_return_time_aligned_trajectory():
    x0 = torch.zeros(2, 1)
    times = torch.linspace(0, 1, 6)

    returned_times, states = EulerIntegrator().integrate(
        lambda t, x: torch.ones_like(x),
        x0,
        times,
        return_trajectory=True,
    )

    torch.testing.assert_close(returned_times, times)
    assert states.shape == (6, 2, 1)
    torch.testing.assert_close(states[-1], torch.ones_like(x0))


def test_integrator_supports_decreasing_time_grid():
    x1 = torch.ones(1, 1)
    times = torch.linspace(1, 0, 5)

    x0 = EulerIntegrator().integrate(
        lambda t, x: torch.ones_like(x),
        x1,
        times,
    )

    torch.testing.assert_close(x0, torch.zeros_like(x1))
