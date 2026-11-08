"""Small end-to-end checks for joint crystal flow-map training and sampling."""

import tempfile
import unittest
from unittest.mock import patch

import lightning as L
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.csp_model.flowmap_csp import Endpoint_Flowmap_CSP
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.rg_vfm import RGVFM
from src.flow_matching.vfm import VFM
from src.lit.litCSPFlowMap import LitCSPFlowMap
from src.manifolds import FlatTorus01
from src.nn.ACSPVNet import ACSPVNet


def make_batch(with_lattice=True):
    graphs = []
    for index, count in enumerate((2, 3, 2, 3)):
        x = torch.linspace(0.1, 0.7, count)[:, None]
        src, dst = torch.meshgrid(torch.arange(count), torch.arange(count), indexing="ij")
        graph = Data(
            x=x, pos=x.clone(), h=torch.full((count,), index, dtype=torch.long),
            edge_index=torch.stack((src.flatten(), dst.flatten())),
        )
        if with_lattice:
            graph.l = torch.linspace(-0.2, 0.3, 6)[None] + index / 10
        graphs.append(graph)
    return Batch.from_data_list(graphs)


def make_module():
    f_flow = VFMFlowMap(RGVFM(FlatTorus01(dim=1), support="extrinsic"))
    l_flow = VFMFlowMap(VFM(dim=6, prior_scale=1.7))
    pi = ACSPVNet(
        hidden_dim=12, time_dim=8, num_layers=2, h_dim=8, num_freqs=3, f_dim=2,
    )
    return LitCSPFlowMap(pi, Endpoint_Flowmap_CSP(f_flow, l_flow))


class LitCSPFlowMapTests(unittest.TestCase):
    def test_training_pair_shares_graph_times_and_converts_intrinsic_coordinates(self):
        module, graph1 = make_module(), make_batch()
        original = graph1.clone()
        graph0 = graph1.clone()
        graph0.x = graph0.pos = torch.linspace(-0.4, 0.8, graph1.num_nodes * 2).reshape(-1, 2)
        graph0.l = graph1.l - 0.5
        t = torch.tensor([[0.2], [0.5], [0.4], [0.7]])
        f1 = module.flow_map.f_flowmap.rg_vfm.to_model_space(graph1.x)

        actual_t, state, target = module.sample_training_pair(graph1, graph0, t=t)

        torch.testing.assert_close(actual_t, t)
        torch.testing.assert_close(state.x, graph0.x + t[graph1.batch] * (f1 - graph0.x))
        torch.testing.assert_close(state.l, graph0.l + t * (graph1.l - graph0.l))
        torch.testing.assert_close(target.x, f1)
        torch.testing.assert_close(target.l, graph1.l)
        torch.testing.assert_close(state.pos, state.x)
        torch.testing.assert_close(graph1.x, original.x)
        torch.testing.assert_close(graph1.l, original.l)

    def test_lattice_pair_and_sampling_use_configured_vfm(self):
        module, graph = make_module(), make_batch()
        self.assertIs(module.l_vfm, module.flow_map.l_flowmap.vfm)
        t = torch.tensor([[0.2], [0.5], [0.4], [0.7]])
        with patch.object(module.l_vfm, "sample_training_pair", wraps=module.l_vfm.sample_training_pair) as pair:
            actual_t, _, _ = module.sample_training_pair(graph, t=t)
        lattice_t = pair.call_args.kwargs["t"]
        self.assertEqual(lattice_t.shape, (graph.num_graphs, 1))
        torch.testing.assert_close(lattice_t, actual_t)
        torch.testing.assert_close(lattice_t, t)

        torch.manual_seed(19)
        module.rg_vfm.sample_prior(
            (graph.num_nodes, module.rg_vfm.model_dim),
            device=graph.x.device, dtype=graph.x.dtype,
        )
        expected_l = 1.7 * torch.randn_like(graph.l)
        torch.manual_seed(19)
        with (
            patch.object(module.l_vfm, "sample_prior", wraps=module.l_vfm.sample_prior) as prior,
            patch.object(module.flow_map, "sample", side_effect=lambda graph0, *args, **kwargs: graph0),
        ):
            result = module.sample(graph)
        prior.assert_called_once()
        torch.testing.assert_close(result.l, expected_l)

    def test_lightning_trains_and_evaluates_with_finite_modality_losses(self):
        torch.manual_seed(9)
        module = make_module()
        parameters = list(module.pi.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        gradients = []
        hooks = [parameter.register_hook(lambda grad: gradients.append(grad.detach().clone())) for parameter in parameters]
        loader = DataLoader(make_batch().to_data_list(), batch_size=4)
        try:
            with tempfile.TemporaryDirectory() as directory:
                trainer = L.Trainer(
                    default_root_dir=directory, accelerator="cpu", devices=1,
                    max_epochs=1, limit_train_batches=1, limit_val_batches=1,
                    limit_test_batches=1, num_sanity_val_steps=0,
                    inference_mode=True, logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False,
                )
                trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
                self.assertEqual(trainer.global_step, 1)
                self.assertGreaterEqual(len(gradients), len(parameters))
                self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, parameters)))
                for stage in ("train", "val"):
                    for loss_name in ("tangent", "lagrange"):
                        for modality in ("f", "l"):
                            value = trainer.callback_metrics[f"{stage}/{loss_name}_loss_{modality}"]
                            self.assertTrue(torch.isfinite(value))
                module.zero_grad(set_to_none=True)
                before = [parameter.detach().clone() for parameter in parameters]
                for metrics in (
                    trainer.validate(module, dataloaders=loader, verbose=False)[0],
                    trainer.test(module, dataloaders=loader, verbose=False)[0],
                ):
                    self.assertTrue(metrics)
                    self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
                for old, new in zip(before, parameters):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                    self.assertIsNone(new.grad)
        finally:
            for hook in hooks:
                hook.remove()

    def test_sample_preserves_template_and_lattice_slices(self):
        module, template = make_module(), make_batch()
        original = template.clone()

        result = module.sample(template, n_steps=1)

        self.assertEqual(result.x.shape, (template.num_nodes, 2))
        self.assertEqual(result.l.shape, (template.num_graphs, 6))
        self.assertTrue(torch.isfinite(result.x).all())
        self.assertTrue(torch.isfinite(result.l).all())
        self.assertFalse(result.x.requires_grad)
        torch.testing.assert_close(result.pos, result.x)
        torch.testing.assert_close(template.x, original.x)
        torch.testing.assert_close(template.h, original.h)
        torch.testing.assert_close(template.l, original.l)
        for graph in result.to_data_list():
            self.assertEqual(graph.l.shape, (1, 6))
            self.assertEqual(graph.x.shape[-1], 2)


if __name__ == "__main__":
    unittest.main()
