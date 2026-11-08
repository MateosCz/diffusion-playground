"""Periodic position features, time conditioning and checkpoint compatibility."""

import unittest
from types import SimpleNamespace

import torch

from src.manifolds import FlatTorus, FlatTorus01
from src.nn.rg_vfm_mlp import RGVFMMLP
from src.nn.scoreNNBlock import sinusoidal_time_embedding


def make_model(**kwargs) -> RGVFMMLP:
    manifold = kwargs.pop("manifold", FlatTorus01(dim=2))
    return RGVFMMLP(
        dim=2,
        x_lifting_dim=16,
        time_embedding_half_dim=4,
        hidden_dim=[32, 32, 16],
        output_dim=2,
        manifold=manifold,
        **kwargs,
    )


class RGVFMMLPTests(unittest.TestCase):
    def test_forward_matches_rg_vfm_model_interface(self):
        model = make_model()
        t = torch.rand(8, 1)
        x_t = torch.rand(8, 2)

        prediction = model(t, x_t)

        self.assertEqual(prediction.shape, x_t.shape)
        prediction.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_forward_accepts_flat_batch_time(self):
        prediction = make_model()(torch.rand(8), torch.rand(8, 2))
        self.assertEqual(prediction.shape, (8, 2))

    def test_position_encoding_uses_manifold_period(self):
        for period in (1.0, 2 * torch.pi):
            with self.subTest(period=period):
                model = make_model(
                    manifold=FlatTorus(dim=2, period=period),
                    position_fourier_bands=3,
                    with_residual_position=False,
                ).eval()
                t = torch.rand(8, 1)
                x_t = torch.rand(8, 2)

                torch.testing.assert_close(
                    model.position_period,
                    torch.full((2,), period),
                )
                torch.testing.assert_close(
                    model(t, x_t),
                    model(t, x_t + period),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_legacy_position_period_keyword_is_ignored(self):
        model = make_model(position_period=2 * torch.pi)
        torch.testing.assert_close(model.position_period, torch.ones(2))

    def test_matching_period_checkpoint_preserves_predictions(self):
        for period in (1.0, 2 * torch.pi):
            with self.subTest(period=period):
                model = make_model(manifold=FlatTorus(dim=2, period=period)).eval()
                restored = make_model(manifold=FlatTorus(dim=2, period=period)).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                t, x_t = torch.rand(8, 1), torch.rand(8, 2)
                torch.testing.assert_close(model(t, x_t), restored(t, x_t))

    def test_legacy_buffer_does_not_override_manifold_period(self):
        model = make_model().eval()
        state = model.state_dict()
        state["position_period"] = torch.full((2,), 2 * torch.pi)
        restored = make_model().eval()
        restored.load_state_dict(state, strict=True)

        torch.testing.assert_close(restored.position_period, state["position_period"])
        t, x_t = torch.rand(8, 1), torch.rand(8, 2)
        torch.testing.assert_close(model(t, x_t), restored(t, x_t))

    def test_position_encoding_can_use_raw_coordinates(self):
        model = make_model(with_sincos_position=False).eval()
        captured_input = []
        handle = model.lifting_layer_x.register_forward_pre_hook(
            lambda module, args: captured_input.append(args[0])
        )
        x_t = torch.rand(8, 2)
        try:
            prediction = model(torch.rand(8, 1), x_t)
        finally:
            handle.remove()

        self.assertEqual(model.lifting_layer_x[0].in_features, 2)
        self.assertEqual(prediction.shape, x_t.shape)
        torch.testing.assert_close(captured_input[0], x_t)

    def test_raw_input_does_not_require_manifold_period(self):
        manifold = SimpleNamespace(project_to_manifold=lambda x: x)
        model = make_model(manifold=manifold, with_sincos_position=False)
        prediction = model(torch.rand(8, 1), torch.rand(8, 2))
        self.assertEqual(prediction.shape, (8, 2))

    def test_sincos_position_remains_the_default(self):
        model = make_model(position_fourier_bands=3)
        self.assertTrue(model.with_sincos_position)
        self.assertEqual(model.lifting_layer_x[0].in_features, 2 * 2 * 3)

    def test_two_times_are_embedded_in_sample_and_column_order(self):
        model = make_model(
            time_input_dim=2, total_time=2.0, time_embedding_scale=3.0,
        )
        t = torch.tensor([[0.1, 0.7], [0.2, 0.8], [0.3, 0.9]])
        captured_input = []
        handle = model.lifting_layer_t.register_forward_pre_hook(
            lambda module, args: captured_input.append(args[0])
        )
        try:
            prediction = model(t, torch.rand(3, 2))
        finally:
            handle.remove()

        expected = torch.cat([
            sinusoidal_time_embedding(
                t[:, i:i + 1] * 1.5, model.time_embedding_half_dim,
            )
            for i in range(2)
        ], dim=-1)
        self.assertEqual(prediction.shape, (3, 2))
        self.assertEqual(model.lifting_layer_t[0].in_features, 16)
        torch.testing.assert_close(captured_input[0], expected)

    def test_two_time_forward_preserves_autograd(self):
        torch.manual_seed(42)
        model = make_model(time_input_dim=2)
        t = torch.rand(8, 2, requires_grad=True)
        x_t = torch.rand(8, 2, requires_grad=True)

        model(t, x_t).square().mean().backward()

        for tensor in (t, x_t, *model.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertTrue((t.grad.abs().sum(dim=0) > 0).all())
        self.assertGreater(x_t.grad.abs().sum().item(), 0)

    def test_forward_rejects_invalid_shapes(self):
        for time_input_dim, t, x_t in (
            (1, torch.rand(3, 1), torch.rand(4, 2)),
            (1, torch.rand(4, 2), torch.rand(4, 2)),
            (1, torch.rand(4, 1), torch.rand(4, 3)),
            (2, torch.rand(4), torch.rand(4, 2)),
            (2, torch.rand(4, 1), torch.rand(4, 2)),
            (2, torch.rand(3, 2), torch.rand(4, 2)),
            (2, torch.rand(4, 2, 1), torch.rand(4, 2)),
        ):
            with self.subTest(time_input_dim=time_input_dim,
                              time_shape=t.shape, position_shape=x_t.shape):
                with self.assertRaises(ValueError):
                    make_model(time_input_dim=time_input_dim)(t, x_t)


if __name__ == "__main__":
    unittest.main()
