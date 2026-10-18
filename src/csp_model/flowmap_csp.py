"""Joint coordinate/lattice flow map on the fixed time interval [0, 1]."""

import torch

from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.vfm import VFM


class Endpoint_Flowmap_CSP:
    """pi(graph, s, t) returns {"f": f_endpoint, "l": l_endpoint}.

    graph is a PyG Batch in model space; s/t have shape (num_graphs, 1).
    Training pairs and prior initialization belong to LitCSPFlowMap.
    """

    def __init__(self, f_flowmap, l_flowmap=None):
        if l_flowmap is None:
            l_flowmap = VFMFlowMap(VFM(dim=6))
        for flow in (f_flowmap, l_flowmap):
            if flow.vfm.total_time != 1.0:
                raise ValueError("CSP flow maps use total_time=1")
        self.f_flowmap = f_flowmap
        self.l_flowmap = l_flowmap

    def Xst(self, graph, s, t, pi):
        """X_st = x_s + (t-s)/(1-s) * (pi_st(x_s)-x_s), for both modalities."""
        prediction = pi(graph, s, t)
        pred_f, pred_l = prediction["f"], prediction["l"]
        alpha = (t - s) / (1 - s)
        f = graph.x + alpha[graph.batch] * (pred_f - graph.x)
        l = graph.l + alpha * (pred_l - graph.l)
        return self._update_graph(graph, f, l)

    def tangent_loss(self, graph1, grapht, t, pi):
        """Return separate endpoint regression losses at s=t."""
        prediction = pi(grapht, t, t)
        pred_f, pred_l = prediction["f"], prediction["l"]
        loss_f = self.f_flowmap.vfm.loss(pred_f, graph1.x, t=t[grapht.batch])
        loss_l = self.l_flowmap.vfm.loss(pred_l, graph1.l, t=t)
        return {"f": loss_f, "l": loss_l}

    def lagrange_loss(self, graph, s, t, pi):
        """Joint time derivative; stop-gradient applies only to pi_tt."""
        def joint_state(target_t):
            graph_st = self.Xst(graph, s, target_t, pi)
            return graph_st.x, graph_st.l

        (f, l), (df_dt, dl_dt) = torch.autograd.functional.jvp(
            joint_state, (t,), (torch.ones_like(t),), create_graph=True,
        )
        prediction = pi(self._update_graph(graph, f, l), t, t)
        pred_f, pred_l = prediction["f"], prediction["l"]
        residual_f = (1 - t)[graph.batch] * df_dt - (pred_f.detach() - f)
        residual_l = (1 - t) * dl_dt - (pred_l.detach() - l)
        return {
            "f": residual_f.square().sum(-1).mean(),
            "l": residual_l.square().sum(-1).mean(),
        }

    @torch.inference_mode()
    def sample(self, graph0, pi, n_steps=1):
        """Advance an already initialized prior graph from 0 to 1."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        times = torch.linspace(0, 1, n_steps + 1, device=graph0.x.device, dtype=graph0.x.dtype)
        graph = graph0
        for s, t in zip(times[:-1], times[1:]):
            s, t = s.expand(graph.l.shape[0], 1), t.expand(graph.l.shape[0], 1)
            graph = self.Xst(graph, s, t, pi)
        return graph

    """Update both modalities without modifying the input graph."""
    def _update_graph(self, graph, f, l):
        updated = graph.clone()
        updated.x = updated.pos = f
        updated.l = l
        return updated


JointFlowMapCSP = Endpoint_Flowmap_CSP

__all__ = ["Endpoint_Flowmap_CSP", "JointFlowMapCSP"]
