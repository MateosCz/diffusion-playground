"""Exercise the flow-map training entry with its datasets and real callbacks."""

from contextlib import chdir
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lightning as L
import torch

from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.lit.litFracFlowMap import LitFracFlowMapNN
from src.litTrain import trainLitFracFlowMap as entry
from src.nn.frac_flowmapNN import ExFlowMapNN


class FlowMapTrainingEntryTests(unittest.TestCase):
    def test_main_trains_checkpoints_and_reloads_for_supported_datasets(self):
        pacman_path = str(Path(__file__).resolve().parents[1] / "data/pacman.npy")
        trainer_class = L.Trainer
        for dataset, configured_dim, expected_dim in (
            ("checkerboard", 1, 1),
            ("checkerboard", 2, 2),
            ("pacman", 1, 2),
        ):
            with self.subTest(dataset=dataset, dim=configured_dim):
                torch.manual_seed(42)
                trainers = []

                def make_trainer(**kwargs):
                    trainer = trainer_class(
                        **kwargs, devices=1, enable_progress_bar=False,
                        enable_model_summary=False,
                    )
                    trainers.append(trainer)
                    return trainer

                small_nn = dict(
                    entry.nn_kwargs, x_lifting_dim=8,
                    time_embedding_half_dim=3, hidden_dim=[12, 8],
                )
                with (
                    tempfile.TemporaryDirectory() as directory,
                    chdir(directory),
                    patch.multiple(
                        entry, dataset_name=dataset, dim=configured_dim,
                        pacman_path=pacman_path, train_size=16, val_size=8,
                        batch_size=8, num_workers=0, n_epoch=1,
                        nn_kwargs=small_nn, generation_eval_every_n_epochs=1,
                        generation_eval_samples=16, generation_eval_steps=2,
                    ),
                    patch.object(entry, "get_default_device", return_value=torch.device("cpu")),
                    patch.object(entry, "WandbLogger", return_value=False),
                    patch.object(entry.wandb, "finish") as finish,
                    patch.object(entry.L, "Trainer", side_effect=make_trainer) as trainer_factory,
                ):
                    entry.main()
                    trainer = trainers[0]
                    model = trainer.lightning_module
                    self.assertIsInstance(model, LitFracFlowMapNN)
                    self.assertIsInstance(model.pi, ExFlowMapNN)
                    self.assertIs(model.flow_map.rg_vfm, model.rg_vfm)
                    self.assertEqual(model.rg_vfm.manifold.intrinsic_dim, expected_dim)
                    self.assertEqual(model.pi.time_input_dim, 2)
                    self.assertTrue(model.pi.with_residual_position)
                    self.assertFalse(model.pi.project_to_manifold)
                    self.assertTrue(model.hparams.nn_kwargs["with_residual_position"])
                    self.assertFalse(model.hparams.nn_kwargs["project_to_manifold"])
                    self.assertEqual(trainer.global_step, 2)
                    self.assertFalse(trainer_factory.call_args.kwargs["inference_mode"])
                    finish.assert_called_once()

                    metrics = trainer.callback_metrics
                    for name in ("val_tangent_loss", "val_lagrange_loss", "val_total_loss"):
                        self.assertTrue(torch.isfinite(metrics[name]))
                    torch.testing.assert_close(
                        metrics["val_total_loss"],
                        metrics["val_tangent_loss"] + metrics["val_lagrange_loss"],
                    )
                    if dataset == "checkerboard":
                        self.assertTrue(torch.isfinite(metrics["val_generated_tv"]))
                        self.assertTrue(torch.isfinite(metrics["val_generated_valid_tile_rate"]))
                        distribution_checkpoint = next(
                            callback for callback in trainer.checkpoint_callbacks
                            if callback.monitor == "val_generated_tv"
                        )
                        self.assertTrue(Path(distribution_checkpoint.best_model_path).is_file())

                    checkpoint = next(
                        callback for callback in trainer.checkpoint_callbacks
                        if callback.monitor == "val_total_loss"
                    )
                    self.assertTrue(Path(checkpoint.best_model_path).is_file())
                    manifold = entry.build_manifold()
                    flow = entry.build_rg_vfm(manifold)
                    restored = LitFracFlowMapNN.load_from_checkpoint(
                        checkpoint.best_model_path, pi=entry.build_model(manifold),
                        rg_vfm=flow, flow_map=VFMFlowMap(flow),
                    )
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], value)
                    samples = restored.sample(4, n_steps=2)
                    self.assertEqual(samples.shape, (4, 2 * expected_dim))
                    self.assertTrue(torch.isfinite(samples).all())


if __name__ == "__main__":
    unittest.main()
