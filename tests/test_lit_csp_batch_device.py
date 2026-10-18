"""Rebatching training graphs must preserve non-CUDA accelerator devices."""

import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data

from src.csp_model.flowmap_csp import JointFlowMapCSP
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.rg_vfm import RGVFM
from src.lit.litCSPFlowMap import LitCSPFlowMap
from src.manifolds import FlatTorus01
from src.nn.ACSPVNet import Ambient_CSPVNet


def make_module():
    model = Ambient_CSPVNet(
        hidden_dim=8, time_dim=4, num_layers=1, h_dim=8, num_freqs=2,
    )
    flow = JointFlowMapCSP(VFMFlowMap(RGVFM(
        FlatTorus01(dim=3), support="extrinsic", max_loss_weight=100.0,
    )))
    return LitCSPFlowMap(model, flow)


def make_batch():
    graphs = []
    for count in (2, 3):
        positions = torch.rand(count, 3)
        src, dst = torch.meshgrid(torch.arange(count), torch.arange(count), indexing="ij")
        graphs.append(Data(
            x=positions, pos=positions.clone(), h=torch.ones(count, dtype=torch.long),
            l=torch.randn(1, 6), edge_index=torch.stack((src.flatten(), dst.flatten())),
        ))
    return Batch.from_data_list(graphs)


class CSPBatchDeviceTests(unittest.TestCase):
    def test_rebuilt_subbatches_keep_all_graph_tensors_on_input_device(self):
        # Like MPS, meta tensors are not CUDA, so PyG recreates batch/ptr on CPU.
        batch, module = make_batch().to("meta"), make_module()

        def check_pair(graph):
            for name in ("x", "pos", "h", "l", "edge_index", "batch", "ptr"):
                self.assertEqual(graph[name].device, batch.x.device, name)
            return torch.empty(graph.num_graphs, 1, device="meta"), graph, graph

        losses = {"f": torch.tensor(0.0), "l": torch.tensor(0.0)}
        with (
            patch.object(module, "sample_training_pair", side_effect=check_pair) as pair,
            patch.object(module.flow_map, "tangent_loss", return_value=losses),
            patch.object(module.flow_map, "lagrange_loss", return_value=losses),
            patch.object(module, "log_dict"),
        ):
            for stage in ("train", "val", "test"):
                with self.subTest(stage=stage):
                    module._shared_step(batch, stage)
            self.assertEqual(pair.call_count, 6)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS hardware unavailable")
    def test_real_training_step_and_backward_on_mps(self):
        torch.manual_seed(47)
        module, batch = make_module().to("mps"), make_batch().to("mps")
        with patch.object(module, "log_dict"):
            loss = module.training_step(batch)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [parameter.grad for parameter in module.pi.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
