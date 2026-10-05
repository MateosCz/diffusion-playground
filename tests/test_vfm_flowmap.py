"""Affine endpoint flow maps and their differentiable Lagrangian loss."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.rg_vfm import RGVFM
from src.manifolds import FlatTorus01
from src.nn.frac_flowmapNN import ExFlowMapNN, InFlowMapNN


class TimeDependentEndpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([0.4, 1.3], dtype=torch.float64))
        self.register_buffer("source", torch.tensor([0.3, -0.2], dtype=torch.float64))
        self.register_buffer("target", torch.tensor([0.5, 0.4], dtype=torch.float64))

    def forward(self, x, s, t):
        return self.theta * x + s * self.source + t.square() * self.target


class ConstantEndpoint(nn.Module):
    def forward(self, x, s, t):
        return torch.full_like(x, 0.75)


def make_flow(total_time=2.0, support="intrinsic"):
    return VFMFlowMap(RGVFM(FlatTorus01(dim=2), total_time=total_time, support=support))


class VFMFlowMapTests(unittest.TestCase):
    def test_lagrange_loss_stops_endpoint_gradient_but_retains_state_gradient(self):
        flow = make_flow()
        pi = TimeDependentEndpoint()
        x = torch.tensor([[0.2, 0.3], [0.7, -0.4], [1.2, 0.1]], dtype=torch.float64)
        s = torch.tensor([[0.1], [0.4], [0.8]], dtype=x.dtype)
        t = torch.tensor([[0.6], [1.1], [1.5]], dtype=x.dtype)
        total_time = flow.rg_vfm.total_time
        theta = pi.theta.detach()
        alpha = (t - s) / (total_time - s)
        delta = (theta - 1) * x + s * pi.source + t.square() * pi.target
        x_st = x + alpha * delta
        derivative = (delta + (t - s) * 2 * t * pi.target) / (total_time - s)
        endpoint_displacement = (theta - 1) * x_st + t * pi.source + t.square() * pi.target
        residual = derivative * (total_time - t) - endpoint_displacement
        expected_loss = residual.square().sum(dim=-1).mean()
        # Detach only pi(x_st, t, t); the subtracted x_st remains differentiable.
        residual_theta = (total_time - t) * x / (total_time - s) + alpha * x
        expected_gradient = (2 * residual * residual_theta).mean(dim=0)
        detached_velocity_gradient = (2 * residual * (total_time - t) * x / (total_time - s)).mean(dim=0)
        self.assertFalse(torch.allclose(expected_gradient, detached_velocity_gradient))

        loss = flow.lagrange_loss(x, s, t, pi)
        self.assertEqual(loss.ndim, 0)
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        torch.testing.assert_close(pi.theta.grad, expected_gradient)

    def test_constant_endpoint_has_zero_loss_including_diagonal(self):
        flow = make_flow()
        x = torch.tensor([[0.2, 0.4], [0.7, 0.9], [0.6, 0.1]], dtype=torch.float64)
        s = torch.tensor([0.0, 0.4, 0.8], dtype=x.dtype)
        for t in (s, torch.tensor([0.5, 1.0, 1.8], dtype=x.dtype)):
            with self.subTest(diagonal=torch.equal(s, t)):
                loss = flow.lagrange_loss(x, s, t, ConstantEndpoint())
                torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-24, rtol=0)

    def test_scalar_flat_and_column_times_give_the_same_batched_result(self):
        flow = make_flow()
        pi = TimeDependentEndpoint()
        x = torch.tensor([[0.2, 0.3], [0.7, -0.4], [1.2, 0.1]], dtype=torch.float64)
        source_times = (0.2, torch.tensor(0.2, dtype=x.dtype), torch.full((3,), 0.2, dtype=x.dtype), torch.full((3, 1), 0.2, dtype=x.dtype))
        target_times = (0.9, torch.tensor(0.9, dtype=x.dtype), torch.full((3,), 0.9, dtype=x.dtype), torch.full((3, 1), 0.9, dtype=x.dtype))
        expected_map = flow.Xst(x, source_times[-1], target_times[-1], pi)
        expected_loss = flow.lagrange_loss(x, source_times[-1], target_times[-1], pi)
        for s in source_times:
            for t in target_times:
                with self.subTest(s_shape=torch.as_tensor(s).shape, t_shape=torch.as_tensor(t).shape):
                    torch.testing.assert_close(flow.Xst(x, s, t, pi), expected_map)
                    torch.testing.assert_close(flow.lagrange_loss(x, s, t, pi), expected_loss)

    def test_affine_map_uses_total_time_and_reaches_the_terminal_endpoint(self):
        flow = make_flow(total_time=2.0)
        pi = ConstantEndpoint()
        x = torch.tensor([[0.2, 0.3], [0.7, -0.4], [1.2, 0.1]], dtype=torch.float64)
        s = torch.tensor([0.1, 0.4, 0.8], dtype=x.dtype)
        t = torch.tensor([0.6, 1.1, 1.5], dtype=x.dtype)
        expected = x + ((t - s) / (2.0 - s))[:, None] * (0.75 - x)
        torch.testing.assert_close(flow.Xst(x, s, t, pi), expected)
        torch.testing.assert_close(flow.Xst(x, s, s, pi), x)
        torch.testing.assert_close(flow.Xst(x, s, 2.0, pi), torch.full_like(x, 0.75))

    def test_invalid_time_order_and_shapes_are_rejected(self):
        flow = make_flow()
        pi = ConstantEndpoint()
        x = torch.rand(3, 2)
        s = torch.tensor([0.1, 0.4, 0.8])
        for method in (flow.Xst, flow.lagrange_loss):
            for source, target in (
                (s, torch.tensor([0.2, 0.1, 0.9])),
                (torch.rand(2), 1.0),
            ):
                with self.subTest(method=method.__name__, source=source, target=target):
                    with self.assertRaises(ValueError):
                        method(x, source, target, pi)

    def test_intrinsic_and_extrinsic_networks_backpropagate_through_loss(self):
        torch.manual_seed(9)
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                flow = make_flow(total_time=1.5, support="extrinsic" if extrinsic else "intrinsic")
                options = dict(
                    manifold=flow.rg_vfm.manifold, x_lifting_dim=8,
                    time_embedding_half_dim=3, hidden_dim=[12, 8], total_time=1.5,
                )
                pi = ExFlowMapNN(**options) if extrinsic else InFlowMapNN(dim=2, output_dim=2, **options)
                x = torch.full((3, pi.dim), 0.3)
                s, t = torch.tensor([0.05, 0.15, 0.3]), torch.tensor([0.4, 0.6, 0.8])
                loss = flow.lagrange_loss(x, s, t, pi)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in pi.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(pi.output_layer.weight.grad.abs().sum().item(), 0)

    def test_sample_follows_uniform_steps_with_one_prediction_per_interval(self):
        flow, pi = make_flow(), TimeDependentEndpoint()
        n_steps = 3
        grid = torch.linspace(0.0, 2.0, n_steps + 1, dtype=pi.theta.dtype)
        initial = torch.tensor([[0.2, 0.3], [0.7, -0.4]], dtype=pi.theta.dtype)
        expected_states = [initial]
        with torch.no_grad():
            for s, t in zip(grid[:-1], grid[1:]):
                x = expected_states[-1]
                endpoint = pi.theta * x + s * pi.source + t.square() * pi.target
                expected_states.append(x + (t - s) * (endpoint - x) / (2.0 - s))
        calls = []
        handle = pi.register_forward_pre_hook(
            lambda module, args: calls.append(tuple(value.clone() for value in args))
        )
        try:
            with patch.object(flow.rg_vfm, "sample_prior", return_value=initial) as prior:
                samples = flow.sample(pi, n_steps=n_steps, num_samples=2)
        finally:
            handle.remove()
        prior.assert_called_once_with((2, 2), device=pi.theta.device, dtype=pi.theta.dtype)
        self.assertEqual(len(calls), n_steps)
        for index, (x, s, t) in enumerate(calls):
            torch.testing.assert_close(x, expected_states[index])
            torch.testing.assert_close(s, torch.full((2, 1), grid[index].item(), dtype=x.dtype))
            torch.testing.assert_close(t, torch.full((2, 1), grid[index + 1].item(), dtype=x.dtype))
        torch.testing.assert_close(samples, expected_states[-1])

    def test_sample_uses_prior_representation_and_model_dtype_without_gradients(self):
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                flow = make_flow(support="extrinsic" if extrinsic else "intrinsic")
                options = dict(
                    manifold=flow.rg_vfm.manifold, x_lifting_dim=8,
                    time_embedding_half_dim=3, hidden_dim=[12, 8], total_time=2.0,
                )
                pi = (ExFlowMapNN(**options) if extrinsic else InFlowMapNN(dim=2, output_dim=2, **options)).double()
                parameter = next(pi.parameters())
                grad_modes = []
                handle = pi.register_forward_pre_hook(
                    lambda module, args: grad_modes.append(torch.is_grad_enabled())
                )
                try:
                    with patch.object(flow.rg_vfm, "sample_prior", wraps=flow.rg_vfm.sample_prior) as prior:
                        samples = flow.sample(pi, n_steps=2, num_samples=5)
                finally:
                    handle.remove()
                prior.assert_called_once_with((5, pi.dim), device=parameter.device, dtype=parameter.dtype)
                self.assertEqual(samples.shape, (5, pi.dim))
                self.assertEqual(samples.dtype, parameter.dtype)
                self.assertEqual(samples.device, parameter.device)
                self.assertTrue(torch.isfinite(samples).all())
                self.assertFalse(samples.requires_grad)
                self.assertIsNone(samples.grad_fn)
                self.assertEqual(grad_modes, [False, False])

    def test_one_step_sample_returns_the_predicted_endpoint(self):
        flow, pi = make_flow(), TimeDependentEndpoint()
        initial = torch.tensor(
            [[0.2, 0.3], [0.7, -0.4], [1.2, 0.1], [0.6, 0.4]], dtype=pi.theta.dtype
        )
        expected = pi.theta.detach() * initial + 4.0 * pi.target
        with patch.object(flow.rg_vfm, "sample_prior", return_value=initial):
            samples = flow.sample(pi, n_steps=1, num_samples=4)
        torch.testing.assert_close(samples, expected)


if __name__ == "__main__":
    unittest.main()
