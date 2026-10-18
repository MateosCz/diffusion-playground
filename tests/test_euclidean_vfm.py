"""Euclidean endpoint regression for unconstrained lattice parameters."""

import unittest

import torch
from torch import nn

from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.vfm import VFM


class ConstantEndpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.endpoint = nn.Parameter(torch.tensor([-2., 3., -4., 5., -6., 7.], dtype=torch.float64))

    def forward(self, x, s, t):
        return self.endpoint.expand_as(x)


class EuclideanVFMTests(unittest.TestCase):
    def test_training_pair_interpolates_unbounded_lattice_values(self):
        flow = VFM(dim=6)
        x0 = torch.tensor([[-6., -4., -2., 2., 4., 6.]]).repeat(2, 1)
        x1 = -x0
        times = torch.tensor([[0.25], [0.75]])
        t, xt, target = flow.sample_training_pair(x1, x0, t=times)
        self.assertEqual(flow.model_dim, 6)
        self.assertEqual(flow.total_time, 1.)
        torch.testing.assert_close(t, times)
        torch.testing.assert_close(xt, (1 - times) * x0 + times * x1)
        torch.testing.assert_close(target, x1)
        self.assertLess(xt.min().item(), 0)
        self.assertGreater(xt.max().item(), 1)

    def test_gaussian_prior_preserves_shape_dtype_and_scale(self):
        flow = VFM(dim=6, prior_scale=2.5)
        reference = torch.empty(4, 6, dtype=torch.float64)
        torch.manual_seed(17)
        expected = 2.5 * torch.randn_like(reference)
        for shape_or_like in ((4, 6), reference):
            with self.subTest(tensor=isinstance(shape_or_like, torch.Tensor)):
                torch.manual_seed(17)
                actual = flow.sample_prior(shape_or_like, dtype=torch.float64)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.device, reference.device)
                self.assertEqual(actual.dtype, reference.dtype)

    def test_loss_sums_features_and_has_expected_time_weighted_gradient(self):
        pred = torch.arange(12, dtype=torch.float64).reshape(2, 6).requires_grad_()
        target = torch.full_like(pred, -2.)
        times = torch.tensor([[0.25], [0.5]], dtype=pred.dtype)
        weight = (1 - times).pow(-2)
        expected = ((pred.detach() - target).square().sum(-1) * weight[:, 0]).mean()
        loss = VFM(dim=6).loss(pred, target, t=times)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(pred.grad, 2 * (pred.detach() - target) * weight / 2)
        unweighted = VFM(dim=6, is_loss_weighted=False).loss(pred, target, t=times)
        torch.testing.assert_close(unweighted, (pred - target).square().sum(-1).mean())

    def test_euclidean_flowmap_has_zero_constant_endpoint_residual_and_samples(self):
        vfm, pi = VFM(dim=6), ConstantEndpoint()
        flow = VFMFlowMap(vfm)
        self.assertIs(flow.vfm, vfm)
        self.assertIs(flow.rg_vfm, vfm)
        x = torch.arange(18, dtype=pi.endpoint.dtype).reshape(3, 6) - 9
        s = torch.tensor([[0.], [0.2], [0.4]], dtype=x.dtype)
        for t in (s, s + 0.3):
            loss = flow.lagrange_loss(x, s, t, pi)
            torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-24, rtol=0)
        for steps in (1, 4):
            samples = flow.sample(pi, n_steps=steps, num_samples=3)
            torch.testing.assert_close(samples, pi.endpoint.expand(3, 6))
            self.assertFalse(samples.requires_grad)

    def test_invalid_training_times_and_feature_shapes_are_rejected(self):
        flow, x = VFM(dim=6), torch.zeros(2, 6)
        for t in (-0.1, 1.):
            with self.subTest(t=t), self.assertRaises(ValueError):
                flow.sample_training_pair(x, t=t)
        for shape in ((2, 5), (2, 2, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                flow.sample_training_pair(torch.zeros(shape))
        with self.assertRaises(ValueError):
            flow.sample_training_pair(x, torch.zeros(3, 6))
        with self.assertRaises(ValueError):
            flow.loss(x, torch.zeros(2, 5))
        with self.assertRaises(ValueError):
            flow.loss(x, x, t=1.)


if __name__ == "__main__":
    unittest.main()
