"""Checkerboard metrics and small 1D training runs without external loggers."""

import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.dataLib.synthetic import Checkerboard_Dataset
from src.flow_matching import RFM, RGVFM
from src.lit.checkerboard_generation_metrics import (
    CheckerboardGenerationMetrics,
    checkerboard_distribution_metrics,
)
from src.lit.litRFMMLP import LitRFMMLP
from src.lit.litRGVFMMLP import LitRGVFMMLP
from src.litTrain.evalFlatTorus2D import build_model_from_checkpoint
from src.manifolds import FlatTorus01
from src.nn.rfm_mlp import RFMMLP
from src.nn.rg_vfm_mlp import EX_RGVFMMLP, RGVFMMLP


class CheckerboardMetricsTests(unittest.TestCase):
    def test_valid_bins_and_uniform_baselines_in_one_and_two_dimensions(self):
        for dim in (1, 2):
            for num_rows in (4, 6):
                with self.subTest(dim=dim, num_rows=num_rows):
                    bins = 4 * num_rows
                    centers = (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
                    grid = torch.meshgrid(*([centers] * dim), indexing="ij")
                    points = torch.stack([axis.flatten() for axis in grid], dim=-1)
                    tile_parity = torch.floor(points * num_rows).long().sum(dim=-1) % 2
                    ideal = checkerboard_distribution_metrics(
                        points[tile_parity == 0], num_rows=num_rows, bins=bins,
                    )
                    baseline = checkerboard_distribution_metrics(
                        points, num_rows=num_rows, bins=bins,
                    )
                    torch.testing.assert_close(ideal["valid_tile_rate"], torch.tensor(1.0))
                    torch.testing.assert_close(ideal["histogram_tv"], torch.tensor(0.0))
                    torch.testing.assert_close(baseline["valid_tile_rate"], torch.tensor(0.5))
                    torch.testing.assert_close(baseline["histogram_tv"], torch.tensor(0.5))


class CheckerboardOneDimensionalTrainingTests(unittest.TestCase):
    def test_training_generation_and_distribution_checkpoint(self):
        for method in ("rgvfm_intrinsic", "rgvfm_extrinsic", "rfm"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                manifold = FlatTorus01(dim=1)
                nn_kwargs = dict(
                    dim=1, output_dim=1, x_lifting_dim=8,
                    time_embedding_half_dim=2, hidden_dim=[8, 8],
                    position_period=1.0,
                )
                if method == "rfm":
                    flow = RFM(manifold)
                    model = RFMMLP(manifold=manifold, **nn_kwargs)
                    module = LitRFMMLP(
                        model=model, rfm=flow, flow_kwargs={},
                        nn_kwargs=nn_kwargs, batch_size=8,
                    )
                else:
                    support = method.removeprefix("rgvfm_")
                    flow_kwargs = dict(
                        support=support, ambient_metric="geodesic",
                        max_velocity_scale=None, max_loss_weight=100.0,
                    )
                    flow = RGVFM(manifold, **flow_kwargs)
                    if support == "extrinsic":
                        nn_kwargs.update(dim=2, output_dim=2, with_sincos_position=False)
                        model = EX_RGVFMMLP(manifold=manifold, **nn_kwargs)
                    else:
                        model = RGVFMMLP(manifold=manifold, **nn_kwargs)
                    module = LitRGVFMMLP(
                        model=model, rg_vfm=flow, flow_kwargs={},
                        rg_vfm_kwargs=flow_kwargs, nn_kwargs=nn_kwargs,
                        experiment_name_timestamp="checkerboard_1d_test", batch_size=8,
                    )

                dataset = Checkerboard_Dataset(num_rows=6, dim=1, dataset_size=16, seed=42)
                loader = DataLoader(dataset, batch_size=8)
                checkpoint_callback = ModelCheckpoint(
                    dirpath=directory, filename="distribution-{epoch}",
                    monitor="val_generated_tv", mode="min", save_top_k=1,
                    every_n_epochs=1, save_on_train_epoch_end=False,
                )
                trainer = L.Trainer(
                    default_root_dir=directory, accelerator="cpu", max_epochs=1,
                    limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
                    logger=False, enable_progress_bar=False, enable_model_summary=False,
                    callbacks=[
                        CheckerboardGenerationMetrics(
                            num_rows=6, bins=24, n_samples=16, n_steps=4, every_n_epochs=1,
                        ),
                        checkpoint_callback,
                    ],
                )
                trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
                self.assertEqual(trainer.global_step, 2)
                for name in ("train/loss", "val_loss", "val_generated_tv", "val_generated_valid_tile_rate"):
                    self.assertTrue(torch.isfinite(trainer.callback_metrics[name]))
                self.assertTrue(0 <= trainer.callback_metrics["val_generated_tv"] <= 1)
                path = Path(checkpoint_callback.best_model_path)
                self.assertTrue(path.is_file())
                torch.testing.assert_close(
                    checkpoint_callback.best_model_score, trainer.callback_metrics["val_generated_tv"],
                )

                checkpoint = torch.load(path, weights_only=True)
                restored = build_model_from_checkpoint(
                    checkpoint, method="rfm" if method == "rfm" else "rgvfm", manifold=manifold,
                )
                x_0 = flow.sample_prior((8, 1))
                t = torch.rand(8, 1)
                torch.testing.assert_close(model(t, x_0), restored(t, x_0))
                samples = flow.sample(restored, x_0, n_steps=4)
                if method != "rfm":
                    samples = flow.to_intrinsic(samples)
                self.assertEqual(samples.shape, (8, 1))
                self.assertTrue(torch.isfinite(samples).all())
                self.assertTrue(((samples >= 0) & (samples < 1)).all())


if __name__ == "__main__":
    unittest.main()
