"""Training and validation regressions for the flow-map Lightning module."""

import tempfile
import unittest
from unittest.mock import patch

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.flow_matching.rg_vfm import RGVFM
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.lit.litFracFlowMap import LitFracFlowMapNN
from src.manifolds import FlatTorus01
from src.nn.frac_flowmapNN import ExFlowMapNN


def make_module(eta):
    manifold = FlatTorus01(dim=2)
    flow = RGVFM(manifold, support="extrinsic", max_loss_weight=100)
    widths = dict(x_lifting_dim=8, time_embedding_half_dim=3, hidden_dim=[12, 8])
    return LitFracFlowMapNN(
        pi=ExFlowMapNN(manifold=manifold, **widths), eta=eta,
        rg_vfm=flow, flow_map=VFMFlowMap(flow), flow_kwargs={},
        rg_vfm_kwargs={}, nn_kwargs=widths,
        experiment_name_timestamp="test", batch_size=8,
    )


class LitFracFlowMapTests(unittest.TestCase):
    def test_both_split_branches_receive_samples_and_loss_backpropagates(self):
        for eta, batch_size, diagonal_size, off_diagonal_size in (
            (0.5, 8, 4, 4),
            (1.0, 8, 7, 1),
            (0.1, 8, 1, 7),
            (0.5, 2, 1, 1),
        ):
            with self.subTest(eta=eta, batch_size=batch_size):
                torch.manual_seed(42)
                module = make_module(eta)
                with (
                    patch.object(module, "log_dict") as log_dict,
                    patch.object(module.flow_map, "tangent_loss", wraps=module.flow_map.tangent_loss) as tangent,
                    patch.object(module.flow_map, "lagrange_loss", wraps=module.flow_map.lagrange_loss) as lagrange,
                ):
                    loss = module.training_step(torch.rand(batch_size, 2))
                tangent.assert_called_once()
                lagrange.assert_called_once()
                self.assertEqual(tangent.call_args.args[0].shape[0], diagonal_size)
                self.assertEqual(lagrange.call_args.args[0].shape[0], off_diagonal_size)
                self.assertTrue(torch.isfinite(loss))
                for value in log_dict.call_args.args[0].values():
                    self.assertTrue(torch.isfinite(value))
                loss.backward()
                for parameter in module.pi.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(module.pi.output_layer.weight.grad.abs().sum().item(), 0)

    def test_actual_batches_smaller_than_two_are_rejected_before_sampling_or_loss(self):
        for step_name in ("training_step", "validation_step", "test_step"):
            for batch_size in (0, 1):
                with self.subTest(step=step_name, batch_size=batch_size):
                    module = make_module(0.5)
                    with (
                        patch.object(module.rg_vfm, "sample_training_pair") as sample_pair,
                        patch.object(module.flow_map, "tangent_loss") as tangent,
                        patch.object(module.flow_map, "lagrange_loss") as lagrange,
                    ):
                        with self.assertRaises(ValueError):
                            getattr(module, step_name)(torch.rand(batch_size, 2))
                    sample_pair.assert_not_called()
                    tangent.assert_not_called()
                    lagrange.assert_not_called()

    def test_lightning_fit_and_evaluation_without_parameter_updates(self):
        torch.manual_seed(9)
        module = make_module(0.5)
        loader = DataLoader(torch.rand(16, 2), batch_size=8)
        with tempfile.TemporaryDirectory() as directory:
            trainer = L.Trainer(
                default_root_dir=directory, accelerator="cpu", devices=1,
                max_epochs=1, limit_train_batches=2, limit_val_batches=1,
                limit_test_batches=1, num_sanity_val_steps=1,
                inference_mode=False, logger=False, enable_checkpointing=False,
                enable_progress_bar=False, enable_model_summary=False,
            )
            trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
            self.assertEqual(trainer.global_step, 2)
            module.zero_grad(set_to_none=True)
            parameters_before = [parameter.detach().clone() for parameter in module.pi.parameters()]
            validation = trainer.validate(module, dataloaders=loader, verbose=False)
            test = trainer.test(module, dataloaders=loader, verbose=False)
            for metrics in (validation[0], test[0]):
                for value in metrics.values():
                    self.assertTrue(torch.isfinite(torch.tensor(value)))
            for before, after in zip(parameters_before, module.pi.parameters()):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertIsNone(after.grad)


if __name__ == "__main__":
    unittest.main()
