"""Extrinsic MLP training, decoding and checkpoint regression checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.flow_matching import RGVFM
from src.lit.checkerboard_generation_metrics import CheckerboardGenerationMetrics
from src.lit.litRGVFMMLP import LitRGVFMMLP
from src.litTrain import trainLitRGVFMMLP as training
from src.litTrain.evalFlatTorus2D import build_model_from_checkpoint, evaluate_checkpoint
from src.manifolds import FlatTorus01
from src.nn.rg_vfm_mlp import EX_RGVFMMLP, RGVFMMLP


WIDTHS = dict(x_lifting_dim=16, time_embedding_half_dim=4, hidden_dim=[16, 16])


class ExtrinsicMLPTests(unittest.TestCase):
    def test_gradients_sampling_and_endpoint_projection(self):
        torch.manual_seed(42)
        for dim in (1, 2):
            for residual in (False, True):
                with self.subTest(dim=dim, residual=residual):
                    manifold = FlatTorus01(dim=dim)
                    flow = RGVFM(manifold, support="extrinsic", max_loss_weight=100)
                    model = EX_RGVFMMLP(manifold, **WIDTHS, with_residual_position=residual)
                    t, x_t, target = flow.sample_training_pair(torch.rand(8, dim))
                    prediction = model(t, x_t)
                    norms = prediction.reshape(8, dim, 2).norm(dim=-1)
                    torch.testing.assert_close(norms, torch.ones_like(norms))
                    flow.loss(prediction, target, t=t).backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    samples = flow.to_intrinsic(flow.sample(model, flow.sample_prior((8, dim)), n_steps=4))
                    self.assertEqual(samples.shape, (8, dim))
                    self.assertTrue(((samples >= 0) & (samples < 1)).all())

    def test_training_config_selects_representation(self):
        manifold = FlatTorus01(dim=2)
        for support in ("intrinsic", "extrinsic"):
            with (patch.dict(training.rg_vfm_kwargs, support=support),
                  patch.dict(training.nn_kwargs, WIDTHS, dim=2, output_dim=2),
                  patch.dict(training.extrinsic_nn_kwargs, with_residual_position=False)):
                model = training.build_model(manifold)
                self.assertEqual(model.dim, 4 if support == "extrinsic" else 2)
                if support == "extrinsic":
                    self.assertIsInstance(model, EX_RGVFMMLP)
                    self.assertFalse(model.with_residual_position)
                    self.assertFalse(model.with_sincos_position)

    def test_intrinsic_checkpoint_round_trip(self):
        model = RGVFMMLP(dim=2, output_dim=2, manifold=FlatTorus01(dim=2),
                         with_residual_position=True, **WIDTHS)
        checkpoint = {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}
        restored = build_model_from_checkpoint(checkpoint, method="rgvfm", manifold=model.manifold)
        t, x = torch.rand(8, 1), torch.rand(8, 2)
        torch.testing.assert_close(model(t, x), restored(t, x))

    def test_legacy_extrinsic_projection_round_trip(self):
        manifold = FlatTorus01(dim=2)
        for projected in (True, False):
            with self.subTest(projected=projected):
                model = EX_RGVFMMLP(manifold, **WIDTHS, project_to_manifold=projected)
                state = {f"model.{k}": v for k, v in model.state_dict().items()
                         if k != "project_to_manifold"}
                # The oldest checkpoints omitted the flag; projection was always on.
                saved_nn = {} if projected else {"project_to_manifold": False}
                checkpoint = {"state_dict": state, "hyper_parameters": {"nn_kwargs": saved_nn}}
                restored = build_model_from_checkpoint(checkpoint, method="rgvfm", manifold=manifold)
                self.assertEqual(bool(restored.project_to_manifold), projected)
                t, x = torch.rand(8, 1), torch.randn(8, 4)
                torch.testing.assert_close(model(t, x), restored(t, x))

    def test_lightning_training_callback_and_saved_generation(self):
        manifold = FlatTorus01(dim=2)
        flow_kwargs = dict(support="extrinsic", max_velocity_scale=None, max_loss_weight=100)
        model = EX_RGVFMMLP(manifold, **WIDTHS)
        module = LitRGVFMMLP(
            model=model, rg_vfm=RGVFM(manifold, **flow_kwargs),
            flow_kwargs={}, rg_vfm_kwargs=flow_kwargs, nn_kwargs=WIDTHS,
            experiment_name_timestamp="test", batch_size=8,
        )
        loader = DataLoader(torch.rand(16, 2), batch_size=8)
        with tempfile.TemporaryDirectory() as directory:
            trainer = L.Trainer(
                default_root_dir=directory, accelerator="cpu", max_epochs=1,
                limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[CheckerboardGenerationMetrics(n_samples=16, n_steps=4, every_n_epochs=1)],
            )
            trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
            self.assertTrue(torch.isfinite(trainer.callback_metrics["val_generated_tv"]))
            path = Path(directory) / "model.ckpt"
            trainer.save_checkpoint(path)
            checkpoint = torch.load(path, weights_only=True)
            restored = build_model_from_checkpoint(checkpoint, method="rgvfm", manifold=manifold)
            t, x = torch.rand(8, 1), torch.randn(8, 4)
            torch.testing.assert_close(model(t, x), restored(t, x))
            with patch("src.litTrain.evalFlatTorus2D.get_default_device", return_value=torch.device("cpu")):
                metrics = evaluate_checkpoint(path, method="rgvfm", n_samples=16, n_steps=4)
            self.assertTrue(0 <= metrics["histogram_tv"] <= 1)


if __name__ == "__main__":
    unittest.main()
