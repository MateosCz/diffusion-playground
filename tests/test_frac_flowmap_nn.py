"""Two-time endpoint prediction and external flow-map construction regressions."""

import io
import unittest

import torch

from src.manifolds import FlatTorus, FlatTorus01
from src.nn.frac_flowmapNN import ExFlowMapNN, InFlowMapNN
from src.nn.rg_vfm_mlp import EX_RGVFMMLP, RGVFMMLP


WIDTHS = dict(x_lifting_dim=12, time_embedding_half_dim=4, hidden_dim=[16, 12])


def make_model(extrinsic=False, **overrides):
    manifold = overrides.pop("manifold", FlatTorus01(dim=2))
    options = dict(WIDTHS, **overrides)
    if extrinsic:
        return ExFlowMapNN(manifold=manifold, **options)
    options.setdefault("dim", manifold.intrinsic_dim)
    options.setdefault("output_dim", manifold.intrinsic_dim)
    return InFlowMapNN(manifold=manifold, **options)


def constant_raw_output(model, values):
    """Leave an unrestricted, constant vector in the network's output bias."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_layer.bias.copy_(torch.as_tensor(values))


class FractionalFlowMapTests(unittest.TestCase):
    def test_defaults_reuse_the_requested_parent_and_position_encoding(self):
        intrinsic = make_model(time_input_dim=2, with_residual_position=True)
        extrinsic = make_model(
            extrinsic=True, time_input_dim=2, with_residual_position=True,
            project_to_manifold=False,
        )
        self.assertIsInstance(intrinsic, RGVFMMLP)
        self.assertIsInstance(extrinsic, EX_RGVFMMLP)
        self.assertTrue(intrinsic.with_sincos_position)
        self.assertEqual(intrinsic.position_fourier_bands, 8)
        torch.testing.assert_close(intrinsic.position_period, torch.ones(2))
        self.assertFalse(extrinsic.with_sincos_position)
        self.assertFalse(extrinsic.project_to_manifold)
        for model in (intrinsic, extrinsic):
            self.assertEqual(model.time_input_dim, 2)
            self.assertTrue(model.with_residual_position)
            self.assertAlmostEqual(float(model.residual_position_scale), 0.01)
            self.assertEqual(
                model.lifting_layer_t[0].in_features,
                4 * model.time_embedding_half_dim,
            )

    def test_defaults_allow_encoding_and_scale_overrides(self):
        intrinsic = make_model(
            manifold=FlatTorus(dim=2, period=2.0),
            with_sincos_position=False,
            position_fourier_bands=3,
            residual_position_scale=0.2,
        )
        self.assertFalse(intrinsic.with_sincos_position)
        self.assertEqual(intrinsic.position_fourier_bands, 3)
        self.assertAlmostEqual(float(intrinsic.residual_position_scale), 0.2)
        torch.testing.assert_close(intrinsic.position_period, torch.full((2,), 2.0))
        extrinsic = make_model(
            extrinsic=True, with_sincos_position=True,
            position_fourier_bands=3, residual_position_scale=0.2,
        )
        self.assertTrue(extrinsic.with_sincos_position)
        self.assertEqual(extrinsic.position_fourier_bands, 3)
        self.assertAlmostEqual(float(extrinsic.residual_position_scale), 0.2)

    def test_endpoint_prediction_does_not_scale_residual_by_time_difference(self):
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                model = make_model(extrinsic=extrinsic, residual_position_scale=0.2)
                raw = torch.tensor([-2.5, 3.5] * (model.dim // 2))
                constant_raw_output(model, raw)
                x = torch.full((3, model.dim), 0.4)
                if extrinsic:
                    x += 2  # Ambient endpoints must remain unprojected.
                s, t = torch.tensor([0.1, 0.8, 0.5]), torch.tensor([0.6, 0.2, 0.5])
                expected = x + 0.2 * raw
                if not extrinsic:
                    expected = model.manifold.project_to_manifold(expected)
                torch.testing.assert_close(model(x, s, t), expected)
                self.assertFalse(torch.allclose(expected[-1], x[-1]))

    def test_accepts_each_combination_of_flat_and_column_times(self):
        for extrinsic in (False, True):
            model = make_model(extrinsic=extrinsic)
            x = torch.rand(3, model.dim)
            s, t = torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.8, 0.7, 0.6])
            expected = model(x, s[:, None], t[:, None])
            for source, target in ((s, t), (s[:, None], t), (s, t[:, None])):
                with self.subTest(extrinsic=extrinsic, s=source.shape, t=target.shape):
                    torch.testing.assert_close(model(x, source, target), expected)

    def test_external_flow_map_has_identity_and_terminal_endpoint(self):
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                model = make_model(
                    extrinsic=extrinsic, residual_position_scale=0.1, total_time=2.0,
                )
                raw = torch.arange(1, model.dim + 1, dtype=torch.float32)
                constant_raw_output(model, raw)
                x = torch.full((3, model.dim), 0.4)
                if extrinsic:
                    x += 2
                s = torch.tensor([0.1, 0.8, 1.5])
                for t in (s, torch.full_like(s, model.total_time)):
                    endpoint = model(x, s, t)
                    alpha = ((t - s) / (model.total_time - s))[:, None]
                    if extrinsic:
                        x_st = x + alpha * (endpoint - x)
                    else:
                        x_st = model.manifold.exp_map(
                            x, alpha * model.manifold.log_map(x, endpoint)
                        )
                    expected = x if t is s else endpoint
                    torch.testing.assert_close(x_st, expected)

    def test_conditioning_retains_absolute_times_as_well_as_time_difference(self):
        torch.manual_seed(17)
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                model = make_model(
                    extrinsic=extrinsic, residual_position_scale=1,
                    time_embedding_scale=10,
                )
                x = torch.full((3, model.dim), 0.4)
                s = torch.full((3,), 0.1)
                t = torch.full((3,), 0.4)
                self.assertFalse(torch.allclose(model(x, s, t), model(x, s + 0.4, t + 0.4)))

    def test_intrinsic_encoding_uses_the_manifold_period(self):
        manifold = FlatTorus(dim=2, period=3.5, center=0.5)
        model = make_model(manifold=manifold).double()
        torch.testing.assert_close(model.position_period, torch.full((2,), 3.5, dtype=torch.float64))
        x = torch.tensor([[0.2, 0.4], [-0.3, 0.7]], dtype=torch.float64)
        s = torch.tensor([0.1, 0.2], dtype=torch.float64)
        t = torch.tensor([0.8, 0.6], dtype=torch.float64)
        features = []
        handle = model.lifting_layer_x.register_forward_pre_hook(
            lambda module, args: features.append(args[0].detach().clone())
        )
        try:
            output = model(x, s, t)
            shifted_output = model(x + manifold.period, s, t)
        finally:
            handle.remove()
        torch.testing.assert_close(features[0], features[1], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(output, shifted_output, atol=1e-10, rtol=1e-10)

    def test_inputs_times_and_network_parameters_receive_gradients(self):
        torch.manual_seed(42)
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                model = make_model(extrinsic=extrinsic)
                x = torch.full((4, model.dim), 0.4, requires_grad=True)
                s = torch.full((4,), 0.2, requires_grad=True)
                t = torch.full((4,), 0.7, requires_grad=True)
                model(x, s, t).square().sum().backward()
                for value in (x, s, t, *model.parameters()):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())
                for value in (x, s, t, model.output_layer.weight):
                    self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_external_flow_map_has_correct_diagonal_time_derivatives(self):
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                model = make_model(
                    extrinsic=extrinsic, residual_position_scale=0.1, total_time=2.0,
                )
                raw = torch.arange(1, model.dim + 1, dtype=torch.float32)
                constant_raw_output(model, raw)
                x = torch.full((3, model.dim), 0.4, requires_grad=True)
                s = torch.full((3,), 0.3, requires_grad=True)
                t = torch.full((3,), 0.3, requires_grad=True)
                endpoint = model(x, s, t)
                alpha = ((t - s) / (model.total_time - s))[:, None]
                if extrinsic:
                    x_st = x + alpha * (endpoint - x)
                else:
                    x_st = model.manifold.exp_map(
                        x, alpha * model.manifold.log_map(x, endpoint)
                    )
                grad_x, grad_s, grad_t = torch.autograd.grad(x_st.sum(), (x, s, t))
                torch.testing.assert_close(grad_x, torch.ones_like(x))
                expected = 0.1 * raw.sum() / (model.total_time - s.detach())
                torch.testing.assert_close(grad_s, -expected)
                torch.testing.assert_close(grad_t, expected)

    def test_rejects_malformed_position_and_time_shapes(self):
        for extrinsic in (False, True):
            model = make_model(extrinsic=extrinsic)
            x, times = torch.rand(3, model.dim), torch.rand(3)
            invalid_times = (torch.tensor(0.2), torch.rand(2), torch.rand(3, 2), torch.rand(3, 1, 1))
            for invalid in invalid_times:
                for source, target in ((invalid, times), (times, invalid)):
                    with self.subTest(extrinsic=extrinsic, s=source.shape, t=target.shape):
                        with self.assertRaises(ValueError):
                            model(x, source, target)
            invalid_positions = (
                torch.tensor(0.2), torch.rand(model.dim),
                torch.rand(3, model.dim + 1), torch.rand(3, 1, model.dim),
            )
            for invalid in invalid_positions:
                with self.subTest(extrinsic=extrinsic, x=invalid.shape):
                    with self.assertRaises(ValueError):
                        model(invalid, times, times)

    def test_state_dict_round_trip_preserves_predictions_and_scale(self):
        for extrinsic in (False, True):
            with self.subTest(extrinsic=extrinsic):
                model = make_model(extrinsic=extrinsic, residual_position_scale=0.17)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = make_model(extrinsic=extrinsic)
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                x = torch.rand(4, model.dim)
                s, t = torch.rand(4), torch.rand(4)
                torch.testing.assert_close(model(x, s, t), restored(x, s, t))
                self.assertAlmostEqual(float(restored.residual_position_scale), 0.17)


if __name__ == "__main__":
    unittest.main()
