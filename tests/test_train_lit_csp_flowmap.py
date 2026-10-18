"""Exercise the MP-20 entry point with real transforms, flows and crystal metrics."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lightning as L
import torch
from torch_geometric.data import Batch

from src.dataLib.realCrystal import CrystalDataset
from src.lit.litCSPFlowMap import LitCSPFlowMap
from src.litTrain import trainLitCSPFlowMap as entry
from src.manifolds import FlatTorus01


DATA_FOLDER = Path(__file__).resolve().parents[1] / "data" / "mp-20"
SMALL_NETWORK = dict(
    hidden_dim=12, time_dim=8, num_layers=1, h_dim=100, num_freqs=2,
    ln=True, smooth=False, pred_h=False, pred_l=True, f_dim=6,
)


@unittest.skipUnless((DATA_FOLDER / "train.pt").exists(), "local MP-20 data is required")
class TrainLitCSPFlowMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the real serialized records, including species and cell parameters.
        cls.records = CrystalDataset.load(DATA_FOLDER / "train.pt")[:2]

    def make_transform(self):
        with patch.object(entry, "data_folder", str(DATA_FOLDER)):
            return entry.build_transform()

    def make_batch(self, transform):
        dataset = CrystalDataset.__new__(CrystalDataset)
        dataset.data = self.records
        dataset.transform = transform
        return Batch.from_data_list([dataset[i] for i in range(2)])

    def make_module(self):
        with patch.object(entry, "nn_kwargs", SMALL_NETWORK):
            return LitCSPFlowMap(entry.build_model(), entry.build_flowmap())

    def test_mp20_fractional_ambient_pairs_and_fixed_species(self):
        batch = self.make_batch(self.make_transform())
        original = batch.clone()
        raw_positions = torch.cat([torch.as_tensor(record["pos"]) for record in self.records]).float()
        raw_species = torch.cat([torch.as_tensor(record["h"]) for record in self.records]).long()
        torch.testing.assert_close(batch.x, raw_positions)
        torch.testing.assert_close(batch.h, raw_species)

        module = self.make_module()
        self.assertIsInstance(module.rg_vfm.manifold, FlatTorus01)
        self.assertEqual(module.rg_vfm.manifold.period, 1.0)
        t = torch.tensor([[0.25], [0.75]])
        actual_t, state, target = module.sample_training_pair(batch, t=t)
        phases = 2 * torch.pi * raw_positions
        expected_ambient = torch.stack((phases.cos(), phases.sin()), dim=-1).flatten(1)
        torch.testing.assert_close(actual_t, t)
        torch.testing.assert_close(target.x, expected_ambient)
        torch.testing.assert_close(target.pos, target.x)
        torch.testing.assert_close(target.x.reshape(-1, 3, 2).square().sum(-1), torch.ones(batch.num_nodes, 3))
        decoded = module.rg_vfm.to_intrinsic(target.x)
        self.assertTrue(((decoded >= 0) & (decoded < 1)).all())
        # MP-20 includes boundary coordinates at 1, which represent the same site as 0.
        fractional_error = torch.remainder(decoded - raw_positions + 0.5, 1) - 0.5
        torch.testing.assert_close(fractional_error, torch.zeros_like(raw_positions), atol=1e-6, rtol=0)
        for graph in (state, target, module.sample(batch, n_steps=1)):
            self.assertEqual(graph.x.shape, (batch.num_nodes, 6))
            self.assertEqual(graph.l.shape, (2, 6))
            torch.testing.assert_close(graph.h, raw_species)
        torch.testing.assert_close(batch.x, original.x)
        torch.testing.assert_close(batch.l, original.l)

    def test_generation_callback_decodes_exact_ambient_endpoints(self):
        transform = self.make_transform()
        batch = self.make_batch(transform)
        module = self.make_module()
        callback = entry.CSPGenerationMetrics(transform, n_steps=1)
        trainer = SimpleNamespace(sanity_checking=False, current_epoch=0)
        logs = {}
        real_metrics_update = callback.metrics.update

        def update_with_fractional_checks(structures, targets):
            for structure, target, record in zip(structures, targets, self.records):
                raw_positions = torch.as_tensor(record["pos"]).double()
                target_positions = torch.as_tensor(target.frac_coords)
                error = torch.remainder(target_positions - raw_positions + 0.5, 1) - 0.5
                torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-6, rtol=0)
                # Generated structures are sorted by species by the KLDM utility.
                generated_positions = torch.as_tensor(structure.frac_coords)
                sorted_target_positions = torch.as_tensor(target.get_sorted_structure().frac_coords)
                error = torch.remainder(generated_positions - sorted_target_positions + 0.5, 1) - 0.5
                torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-6, rtol=0)
            return real_metrics_update(structures, targets)

        def exact_sample(template, n_steps=1):
            self.assertEqual(n_steps, 1)
            return module.flow_map._update_graph(
                template, module.rg_vfm.to_model_space(template.x), template.l,
            )

        with patch.object(module, "sample", side_effect=exact_sample) as sample, patch.object(
            module, "log_dict", side_effect=lambda values, **kwargs: logs.update(values),
        ), patch.object(
            callback.metrics, "update", side_effect=update_with_fractional_checks,
        ):
            callback.on_validation_epoch_start(trainer, module)
            callback.on_validation_batch_end(trainer, module, None, batch, 0)
            callback.on_validation_epoch_end(trainer, module)
        sample.assert_called_once()
        self.assertEqual(float(logs["val/valid"]), 1.0)
        self.assertEqual(float(logs["val/match_rate"]), 1.0)
        self.assertLess(float(logs["val/rmse"]), 1e-5)

    def test_main_trains_validates_samples_and_reloads_checkpoint(self):
        torch.manual_seed(26)
        real_trainer, real_build_model = L.Trainer, entry.build_model
        captured = {}
        gradients = []

        def build_small_model():
            model = real_build_model()
            captured["before"] = [parameter.detach().clone() for parameter in model.parameters()]
            for parameter in model.parameters():
                parameter.register_hook(lambda grad: gradients.append(grad.detach().clone()))
            return model

        def limited_trainer(**kwargs):
            kwargs.update(
                accelerator="cpu", devices=1, max_epochs=1, logger=False,
                limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
                enable_progress_bar=False, enable_model_summary=False,
            )
            trainer = real_trainer(**kwargs)
            captured["trainer"] = trainer
            return trainer

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                with (
                    patch.object(entry, "data_folder", str(DATA_FOLDER)),
                    patch.object(entry, "nn_kwargs", SMALL_NETWORK),
                    patch.object(entry, "batch_size", 2),
                    patch.object(entry, "evaluate_batch_size", 2),
                    patch.object(entry, "num_workers", 0),
                    patch.object(entry, "n_epoch", 1),
                    patch.object(entry, "generation_eval_every_n_epochs", 1),
                    patch.object(entry, "build_model", side_effect=build_small_model),
                    patch.object(entry.L, "Trainer", side_effect=limited_trainer),
                    patch.object(entry, "WandbLogger", return_value=False),
                    patch.object(entry.wandb, "finish"),
                ):
                    entry.main()

                trainer = captured["trainer"]
                module = trainer.lightning_module
                self.assertEqual(trainer.global_step, 1)
                self.assertFalse(trainer.validate_loop.inference_mode)
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)
                self.assertTrue(any(
                    not torch.equal(before, after)
                    for before, after in zip(captured["before"], module.pi.parameters())
                ))
                for metric in (
                    "train/total_loss", "val/total_loss", "val/valid", "val/match_rate",
                ):
                    self.assertTrue(torch.isfinite(trainer.callback_metrics[metric]))
                self.assertIn("val/rmse", trainer.callback_metrics)
                checkpoint = Path(trainer.checkpoint_callback.best_model_path)
                self.assertTrue(checkpoint.is_file())
                with patch.object(entry, "nn_kwargs", SMALL_NETWORK):
                    restored = LitCSPFlowMap.load_from_checkpoint(
                        checkpoint, pi=entry.build_model(), flow_map=entry.build_flowmap(),
                    )
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value)
                sample = restored.sample(self.make_batch(self.make_transform()), n_steps=1)
                self.assertEqual(sample.x.shape[-1], 6)
                self.assertTrue(torch.isfinite(sample.x).all())
                self.assertTrue(torch.isfinite(sample.l).all())
            finally:
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
