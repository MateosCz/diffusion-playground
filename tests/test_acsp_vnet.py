"""Real CSP endpoint network: graph inputs, symmetry and joint flow losses."""

import unittest

import torch
from torch_geometric.data import Batch, Data

from src.csp_model.flowmap_csp import Endpoint_Flowmap_CSP
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.rg_vfm import RGVFM
from src.flow_matching.vfm import VFM
from src.manifolds import FlatTorus, FlatTorus01
from src.nn.ACSPVNet import ACSPVNet


def make_graph(manifold):
    graphs = []
    for count in (2, 3):
        x = manifold.to_ambient(manifold.sample(count, dtype=torch.float64))
        # Extrinsic interpolants and priors need not lie on the unit circles.
        radius = 0.6 + torch.rand(count, manifold.intrinsic_dim, dtype=x.dtype)
        x = x * radius.repeat_interleave(2, dim=-1)
        src, dst = torch.meshgrid(torch.arange(count), torch.arange(count), indexing="ij")
        graphs.append(Data(
            x=x, pos=x.clone(), h=torch.arange(count, dtype=torch.long) + 1,
            l=torch.randn(1, 6, dtype=x.dtype),
            edge_index=torch.stack((src.flatten(), dst.flatten())),
        ))
    return Batch.from_data_list(graphs)


def make_model(**kwargs):
    return ACSPVNet(
        hidden_dim=12, time_dim=8, num_layers=2, h_dim=8, num_freqs=3,
        **kwargs,
    ).double()


def rotate_pairs(x, phases, batch):
    pairs = x.reshape(x.shape[0], -1, 2)
    c, s = phases[batch].cos(), phases[batch].sin()
    return torch.stack((
        c * pairs[..., 0] - s * pairs[..., 1],
        s * pairs[..., 0] + c * pairs[..., 1],
    ), dim=-1).flatten(start_dim=1)


class ACSPVNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.s = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
        self.t = torch.tensor([[0.6], [0.8]], dtype=torch.float64)

    def assert_finite_gradients(self, model):
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)

    def test_graph_forward_predicts_both_endpoints_without_mutating_inputs(self):
        for manifold in (FlatTorus(dim=3), FlatTorus01(dim=1), FlatTorus01(dim=3)):
            with self.subTest(manifold=manifold):
                model = make_model(f_dim=manifold.ambient_dim)
                graph = make_graph(manifold)
                original = graph.clone()
                output = model(graph, self.s, self.t)
                self.assertEqual(set(output), {"f", "l"})
                self.assertEqual(output["f"].shape, graph.x.shape)
                self.assertEqual(output["l"].shape, (2, 6))
                for key, value in original:
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(graph[key], value, rtol=0, atol=0)

    def test_both_time_arguments_and_output_heads_receive_gradients(self):
        graph, model = make_graph(FlatTorus(dim=3)), make_model()
        s, t = self.s.clone().requires_grad_(), self.t.clone().requires_grad_()
        output = model(graph, s, t)
        sum(value.square().sum() for value in output.values()).backward()
        for time in (s, t):
            self.assertIsNotNone(time.grad)
            self.assertTrue(torch.isfinite(time.grad).all())
            self.assertGreater(time.grad.abs().sum().item(), 0)
        self.assert_finite_gradients(model)

    def test_node_permutation_permutes_coordinates_and_preserves_lattice(self):
        graph, model = make_graph(FlatTorus(dim=3)), make_model()
        permutation = torch.tensor([1, 0, 4, 2, 3])
        inverse = torch.argsort(permutation)
        permuted = graph.clone()
        for key in ("x", "pos", "h", "batch"):
            permuted[key] = graph[key][permutation]
        permuted.edge_index = inverse[graph.edge_index]
        expected, actual = model(graph, self.s, self.t), model(permuted, self.s, self.t)
        torch.testing.assert_close(actual["f"], expected["f"][permutation])
        torch.testing.assert_close(actual["l"], expected["l"])

    def test_torus_translation_rotates_each_ambient_pair_equivariantly(self):
        for manifold in (FlatTorus(dim=3), FlatTorus01(dim=3)):
            with self.subTest(manifold=manifold):
                graph, model = make_graph(manifold), make_model()
                phases = torch.tensor([[0.2, -0.4, 0.75], [-0.6, 0.1, 0.33]], dtype=graph.x.dtype)
                shifted = graph.clone()
                shifted.x = shifted.pos = rotate_pairs(graph.x, phases, graph.batch)
                expected, actual = model(graph, self.s, self.t), model(shifted, self.s, self.t)
                torch.testing.assert_close(
                    actual["f"], rotate_pairs(expected["f"], phases, graph.batch),
                    atol=1e-10, rtol=1e-10,
                )
                torch.testing.assert_close(actual["l"], expected["l"], atol=1e-10, rtol=1e-10)

    def test_real_network_supports_joint_flow_losses_backward_and_sampling(self):
        manifold = FlatTorus01(dim=3)
        graph0, model = make_graph(manifold), make_model()
        flow = Endpoint_Flowmap_CSP(
            VFMFlowMap(RGVFM(manifold, support="extrinsic")), VFMFlowMap(VFM(dim=6)),
        )
        graph1 = graph0.clone()
        graph1.x = graph1.pos = manifold.to_ambient(manifold.sample(5, dtype=graph0.x.dtype))
        graph1.l = torch.randn_like(graph0.l)
        graph_s, graph_t = graph0.clone(), graph0.clone()
        for state, times in ((graph_s, self.s), (graph_t, self.t)):
            _, state.x, _ = flow.f_flowmap.vfm.sample_training_pair(
                graph1.x, graph0.x, t=times[graph0.batch],
            )
            state.pos = state.x
            _, state.l, _ = flow.l_flowmap.vfm.sample_training_pair(graph1.l, graph0.l, t=times)

        graph_zero = graph0.clone()
        graph_zero.x = graph_zero.pos = torch.zeros_like(graph0.x)
        for losses in (
            flow.tangent_loss(graph1, graph_t, self.t, model),
            flow.lagrange_loss(graph_s, self.s, self.t, model),
            flow.lagrange_loss(graph_zero, self.s, self.t, model),
        ):
            self.assertEqual(set(losses), {"f", "l"})
            self.assertTrue(all(torch.isfinite(value) and value.requires_grad for value in losses.values()))
            model.zero_grad(set_to_none=True)
            sum(losses.values()).backward()
            self.assert_finite_gradients(model)

        for steps in (1, 3):
            sampled = flow.sample(graph0, model, n_steps=steps)
            self.assertTrue(torch.isfinite(sampled.x).all())
            self.assertTrue(torch.isfinite(sampled.l).all())
            self.assertFalse(sampled.x.requires_grad)
            torch.testing.assert_close(sampled.pos, sampled.x)
            torch.testing.assert_close(sampled.h, graph0.h)


if __name__ == "__main__":
    unittest.main()
