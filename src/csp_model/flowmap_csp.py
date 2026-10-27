"""Joint coordinate/lattice flow map on the fixed time interval [0, 1]."""

import math
import torch
from torch_geometric.nn import global_mean_pool

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
        loss_f = self.f_flowmap.vfm.loss(
            pred_f, graph1.x, t=t[grapht.batch], reduction="none",
        )
        loss_f = global_mean_pool(loss_f, grapht.batch, size=grapht.num_graphs).mean()
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
        with torch.no_grad():
            prediction = pi(self._update_graph(graph, f, l), t, t)
        pred_f1, pred_l1 = prediction["f"], prediction["l"]
        lagrange_f = (1 - t)[graph.batch] * df_dt - (pred_f1 - f).detach()
        lagrange_l = (1 - t) * dl_dt - (pred_l1 - l).detach()
        return {
            "f": global_mean_pool(
                lagrange_f.square().sum(-1), graph.batch, size=graph.num_graphs,
            ).mean(),
            "l": lagrange_l.square().sum(-1).mean(),
        }

    @torch.inference_mode()
    def sample(self, graph0, pi, n_steps=1, t_max_clip=0.95, time_dist="uniform", pc = False, pc_steps=2):
        """Advance an already initialized prior graph from 0 to 1."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        times = torch.linspace(0, t_max_clip, n_steps + 1, device=graph0.x.device, dtype=graph0.x.dtype)
        if time_dist == "quadratic":
            times = times.square()
        elif time_dist == "uniform":
            times = times
        else:
            raise ValueError(f"unknown time distribution: {time_dist}")
        graph = graph0
        for i, (s, t) in enumerate(zip(times[:-1], times[1:])):
            s, t = s.expand(graph.l.shape[0], 1), t.expand(graph.l.shape[0], 1)
            graph = self.Xst(graph, s, t, pi)
            if pc and i == n_steps - 2: # only apply pc if t < t_max_clip
                for _ in range(pc_steps):
                    score = self.score(graph, t, pi)
                    graph = self.langevin_correction_step(graph, score, t, eta=1)

        return graph

    def score(self, graph, t, pi):
        "Score for the flow map at time t"
        endpoint = pi(graph, t, t)
        f_t = graph.x
        l_t = graph.l
        pred_f1 = endpoint["f"]
        pred_l1 = endpoint["l"]

        t_f = t[graph.batch]
        t_l = t

        vt_ft = (pred_f1 - f_t) / (1-t_f)
        vt_lt = (pred_l1 - l_t) / (1-t_l)

        var_f = self.f_flowmap.vfm.ambient_prior_scale **2
        var_l = self.l_flowmap.vfm.prior_scale ** 2

        score_f = (t_f * vt_ft - f_t) / ((1-t_f) * var_f) # score of f    
        score_l = (t_l * vt_lt - l_t) / ((1-t_l) * var_l) # score of l

        return {"score_f": score_f, "score_l": score_l}


    @torch.inference_mode()
    def sample_euler(self, graph0, pi, n_steps=100, t_max_clip=0.99, pc = False, pc_steps=2):
        """Integrate the diagonal velocity from 0 to t_max_clip with Euler steps."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if not 0 < t_max_clip < 1:
            raise ValueError("t_max_clip must be in (0, 1)")
        times = torch.linspace(
            0, t_max_clip, n_steps + 1,
            device=graph0.x.device, dtype=graph0.x.dtype,
        )
        graph = graph0
        for start, end in zip(times[:-1], times[1:]):
            dt = end - start
            t = start.expand(graph.l.shape[0], 1)
            prediction = pi(graph, t, t)
            pred_f1 = prediction["f"]
            pred_l1 = prediction["l"]
            vt_f = (pred_f1 - graph.x) / (1 - t)[graph.batch]
            vt_l = (pred_l1 - graph.l) / (1 - t)
            f_next = graph.x + dt * vt_f
            l_next = graph.l + dt * vt_l
            graph = self._update_graph(graph, f_next, l_next)
            t = t + dt
            if pc and (t < t_max_clip).all().item(): # only apply pc if t < t_max_clip
                for _ in range(pc_steps):
                    
                    score = self.score(graph, t, pi)
                    graph = self.langevin_correction_step(graph, score, t, eta=0.02)
        return graph

    
    @torch.inference_mode()
    def langevin_correction_step(self, graph, score, t, eta=0.02):
        "one step of Langevin correction for the flow map at time t"
        score_f = score["score_f"]
        score_l = score["score_l"]

        f = graph.x
        l = graph.l

        tau_l = eta * (1 - t).square() * self.l_flowmap.vfm.prior_scale**2
        tau_f = (
            eta * (1 - t).square()
            * self.f_flowmap.vfm.ambient_prior_scale**2
        )[graph.batch]


        f_next = f + tau_f * score_f + (2 * tau_f).sqrt() * torch.randn_like(f)
        l_next = l + tau_l * score_l + (2 * tau_l).sqrt() * torch.randn_like(l)
        graph = self._update_graph(graph, f_next, l_next)
        return graph


    """Update both modalities without modifying the input graph."""
    def _update_graph(self, graph, f, l):
        updated = graph.clone()
        updated.x = updated.pos = f
        updated.l = l
        return updated


JointFlowMapCSP = Endpoint_Flowmap_CSP

__all__ = ["Endpoint_Flowmap_CSP", "JointFlowMapCSP"]
