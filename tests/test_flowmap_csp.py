"""The minimal joint coordinate/lattice flow map uses graph times in [0, 1]."""

import unittest

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from src.csp_model.flowmap_csp import Endpoint_Flowmap_CSP
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.rg_vfm import RGVFM
from src.flow_matching.vfm import VFM
from src.manifolds import FlatTorus01


def make_flow(total_time=1.0):
    return Endpoint_Flowmap_CSP(VFMFlowMap(RGVFM(
        FlatTorus01(dim=1), total_time=total_time, support="extrinsic",
    )))


def make_graph():
    graphs = []
    for index, count in enumerate((2, 3)):
        x = torch.arange(count * 2, dtype=torch.float64).reshape(count, 2) / 10
        nodes = torch.arange(count)
        edges = torch.stack((nodes, nodes.roll(-1)))
        graphs.append(Data(
            x=x, pos=x.clone(), l=torch.arange(6, dtype=x.dtype)[None] / 10 + index,
            h=torch.full((count, 1), index), edge_index=edges,
            edge_attr=x[edges[0]] - x[edges[1]],
        ))
    return Batch.from_data_list(graphs)


class JointEndpoint(nn.Module):
    """Both endpoints depend on both modalities and both time arguments."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([0.4, -0.3, 0.6, 0.2], dtype=torch.float64))

    def predict(self, f, l, batch, s, t):
        pooled = torch.zeros((l.shape[0], f.shape[1]), dtype=f.dtype)
        pooled = pooled.index_add(0, batch, f) / torch.bincount(batch)[:, None]
        return (
            self.theta[0] * f + self.theta[1] * l[batch, :2] + s[batch] + t[batch].square(),
            self.theta[2] * l + self.theta[3] * pooled.repeat(1, 3) + s + t.square(),
        )

    def forward(self, graph, s, t):
        f, l = self.predict(graph.x, graph.l, graph.batch, s, t)
        return {"f": f, "l": l}


class ConstantEndpoint(nn.Module):
    def forward(self, graph, s, t):
        return {"f": torch.full_like(graph.x, 0.75), "l": torch.full_like(graph.l, -0.25)}


class JointFlowMapTests(unittest.TestCase):
    def assert_graph_unchanged(self, graph, original):
        for name, value in original:
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(graph[name], value, rtol=0, atol=0)

    def assert_state_consistent(self, graph):
        torch.testing.assert_close(graph.pos, graph.x)

    def test_lattice_defaults_to_six_dimensional_euclidean_vfm(self):
        flow = make_flow()
        self.assertIsInstance(flow.l_flowmap.vfm, VFM)
        self.assertEqual(flow.l_flowmap.vfm.model_dim, 6)
        self.assertEqual(flow.l_flowmap.vfm.total_time, 1.0)

    def test_affine_map_uses_one_time_per_graph_and_preserves_input(self):
        flow, graph, pi = make_flow(), make_graph(), JointEndpoint()
        original = graph.clone()
        s = torch.tensor([[0.2], [0.4]], dtype=graph.x.dtype)
        t = torch.tensor([[0.6], [0.9]], dtype=graph.x.dtype)
        endpoint = pi(graph, s, t)
        endpoint_f, endpoint_l = endpoint["f"], endpoint["l"]
        alpha = (t - s) / (1 - s)

        result = flow.Xst(graph, s, t, pi)

        self.assertIsNot(result, graph)
        torch.testing.assert_close(result.x, graph.x + alpha[graph.batch] * (endpoint_f - graph.x))
        torch.testing.assert_close(result.l, graph.l + alpha * (endpoint_l - graph.l))
        torch.testing.assert_close(result.h, graph.h)
        torch.testing.assert_close(result.edge_index, graph.edge_index)
        self.assert_state_consistent(result)
        self.assert_graph_unchanged(graph, original)
        diagonal = flow.Xst(graph, s, s, pi)
        terminal = flow.Xst(graph, s, torch.ones_like(s), pi)
        torch.testing.assert_close(diagonal.x, graph.x)
        torch.testing.assert_close(diagonal.l, graph.l)
        endpoint = pi(graph, s, torch.ones_like(s))
        endpoint_f, endpoint_l = endpoint["f"], endpoint["l"]
        torch.testing.assert_close(terminal.x, endpoint_f)
        torch.testing.assert_close(terminal.l, endpoint_l)

    def test_tangent_loss_weights_nodes_and_graph_lattices(self):
        flow, target, pi = make_flow(), make_graph(), JointEndpoint()
        state = target.clone()
        state.x = state.pos = state.x + 0.15
        state.l = state.l - 0.2
        t = torch.tensor([[0.3], [0.7]], dtype=state.x.dtype)
        prediction = pi(state, t, t)
        pred_f, pred_l = prediction["f"], prediction["l"]

        losses = flow.tangent_loss(target, state, t, pi)

        self.assertEqual(set(losses), {"f", "l"})
        expected_f = flow.f_flowmap.rg_vfm.loss(pred_f, target.x, t=t[state.batch])
        expected_l = ((pred_l - target.l) / (1 - t)).square().sum(-1).mean()
        torch.testing.assert_close(losses["f"], expected_f)
        torch.testing.assert_close(losses["l"], expected_l)

    def test_lagrange_joint_jvp_and_only_diagonal_endpoint_stop_gradient(self):
        flow, graph, pi = make_flow(), make_graph(), JointEndpoint()
        s = torch.tensor([[0.1], [0.4]], dtype=graph.x.dtype)
        t = torch.tensor([[0.7], [0.8]], dtype=graph.x.dtype)
        endpoint = pi(graph, s, t)
        endpoint_f, endpoint_l = endpoint["f"], endpoint["l"]
        alpha = (t - s) / (1 - s)
        f = graph.x + alpha[graph.batch] * (endpoint_f - graph.x)
        l = graph.l + alpha * (endpoint_l - graph.l)
        df = (endpoint_f - graph.x + ((t - s) * 2 * t)[graph.batch]) / (1 - s)[graph.batch]
        dl = (endpoint_l - graph.l + (t - s) * 2 * t) / (1 - s)
        diagonal_f, diagonal_l = pi.predict(f, l, graph.batch, t, t)
        expected = {
            "f": ((1 - t)[graph.batch] * df - (diagonal_f.detach() - f)).square().sum(-1).mean(),
            "l": ((1 - t) * dl - (diagonal_l.detach() - l)).square().sum(-1).mean(),
        }
        losses = flow.lagrange_loss(graph, s, t, pi)
        self.assertEqual(set(losses), {"f", "l"})
        for name, cross_head in (("f", 1), ("l", 3)):
            expected_grad = torch.autograd.grad(expected[name], pi.theta, retain_graph=True)[0]
            actual_grad = torch.autograd.grad(losses[name], pi.theta, retain_graph=True)[0]
            torch.testing.assert_close(losses[name], expected[name])
            torch.testing.assert_close(actual_grad, expected_grad)
            self.assertGreater(expected_grad[cross_head].abs().item(), 0)

    def test_constant_endpoint_lagrange_is_zero_and_validation_works_without_grad(self):
        flow, graph, pi = make_flow(), make_graph(), JointEndpoint()
        s = torch.tensor([[0.1], [0.4]], dtype=graph.x.dtype)
        t = torch.tensor([[0.7], [0.8]], dtype=graph.x.dtype)
        for end in (s, t):
            for loss in flow.lagrange_loss(graph, s, end, ConstantEndpoint()).values():
                torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-24, rtol=0)
        training = flow.lagrange_loss(graph, s, t, pi)
        with torch.no_grad():
            validation = flow.lagrange_loss(graph, s, t, pi)
        for name in ("f", "l"):
            torch.testing.assert_close(validation[name], training[name].detach())
            self.assertFalse(validation[name].requires_grad)

    def test_sample_advances_given_prior_with_coupled_states_and_preserves_input(self):
        flow, graph, pi = make_flow(), make_graph(), JointEndpoint()
        original = graph.clone()
        expected_f, expected_l = graph.x, graph.l
        times = torch.linspace(0, 1, 4, dtype=graph.x.dtype)
        with torch.no_grad():
            for start, end in zip(times[:-1], times[1:]):
                s, t = start.expand(2, 1), end.expand(2, 1)
                endpoint_f, endpoint_l = pi.predict(expected_f, expected_l, graph.batch, s, t)
                alpha = (end - start) / (1 - start)
                expected_f = expected_f + alpha * (endpoint_f - expected_f)
                expected_l = expected_l + alpha * (endpoint_l - expected_l)
        result = flow.sample(graph, pi, n_steps=3)
        torch.testing.assert_close(result.x, expected_f)
        torch.testing.assert_close(result.l, expected_l)
        self.assert_state_consistent(result)
        self.assertFalse(result.x.requires_grad)
        self.assert_graph_unchanged(graph, original)
        one_step = flow.sample(graph, pi)
        endpoint = pi(graph, torch.zeros((2, 1)), torch.ones((2, 1)))
        endpoint_f, endpoint_l = endpoint["f"], endpoint["l"]
        torch.testing.assert_close(one_step.x, endpoint_f)
        torch.testing.assert_close(one_step.l, endpoint_l)

    def test_total_time_is_fixed_to_one_for_both_modalities(self):
        with self.assertRaises(ValueError):
            make_flow(total_time=2.0)
        lattice_flow = VFMFlowMap(RGVFM(FlatTorus01(dim=6), total_time=2.0))
        with self.assertRaises(ValueError):
            Endpoint_Flowmap_CSP(make_flow().f_flowmap, lattice_flow)


if __name__ == "__main__":
    unittest.main()
