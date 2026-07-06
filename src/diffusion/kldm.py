import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence, Callable, Tuple
from src.diffusion.sde import VPSDE, BaseSDE, BaseSDEIntegrator, EulerIntegrator, LinearSchedule
from src.distribution import WrappedNormalDistribution
from src.diffusion.tdm import TDMDiffusion, BaseDiffusion
from src.diffusion.continuous import ContinuousDiffusion
from src.dataLib.synthetic import wrap_angle, pos_to_angle, scatter_center
from src.device import module_device
from torch_geometric.data import Data
from tqdm.auto import tqdm


class KLDM(nn.Module):  
    def __init__(
        self, 
        l_diffusion: ContinuousDiffusion, 
        f_diffusion: TDMDiffusion, 
        h_diffusion: Optional[BaseDiffusion] = None,
        total_time: float = 2.0,
        ):
        super().__init__()
        self.l_diffusion = l_diffusion
        self.f_diffusion = f_diffusion
        self.h_diffusion = h_diffusion
        self.total_time = total_time

    def sample_forward(
        self, 
        graph0: Data, 
        # input graph data, x,pos: fractional coordinates, 
        # l: lattice vectors, batch: batch vector, edge_index: edge index, 
        # edge_attr: edge attributes
        t_min: float = 5e-3,
        t_dist_kw: Literal["uniform", "quadratic"] = "uniform",
        shared_time: bool = True,
        return_time: bool = True,
        ):
        # initialize the score and time dictionaries
        score_dict = {}
        t_dict = {}

        # When ``shared_time`` is set (the default), draw a single diffusion time
        # per graph and use it for *both* the lattice and the fractional
        # coordinates. This is required whenever a single-time joint score network
        # (e.g. CSPVNet) predicts the lattice and coordinate scores together.
        shared_ts = None
        if shared_time:
            shared_ts = self._sample_shared_time(graph0, t_dist_kw=t_dist_kw, t_min=t_min)

        # sample the lattice vectors
        l = graph0.l # (num_graph, 6)
        l_expanded = l.unsqueeze(-1) # (num_graph, 6, 1)
        noised_l, score_l, t_l = self.l_diffusion.sample_forward(
            l_expanded, t_dist_kw=t_dist_kw, return_time=True, t_min=t_min, ts=shared_ts,
        )
        noised_l = noised_l.squeeze(-1) # (num_graph, 6)
        score_dict["l"] = score_l
        t_dict["l"] = t_l

        # sample the fractional coordinates
        # sample_forward_graph returns (vts, noised_graph): the second element
        # is a *cloned* PyG batch whose node coords are already set to f_t.
        (noised_v, noised_graph), score_f, t_f = self.f_diffusion.sample_forward_graph(
            graph0, t_dist_kw=t_dist_kw, t_min=t_min, return_time=True, ts=shared_ts,
        )
        score_dict["f"] = score_f
        t_dict["f"] = t_f

        # construct the noised PyG data: write the noised lattice (num_graph, 6)
        # back onto the batch so every sub-graph carries its own (6,) lattice.
        noised_graph = self.update_lattice(noised_graph, noised_l)

        return (noised_v, noised_graph), score_dict, t_dict

    def _sample_shared_time(
        self,
        graph0: Data,
        t_dist_kw: Literal["uniform", "quadratic"] = "uniform",
        t_min: float = 5e-3,
    ) -> torch.Tensor:
        """Draw one diffusion time per graph, shape (num_graph, 1)."""
        device = graph0.x.device
        dtype = graph0.x.dtype
        num_graphs = getattr(graph0, "num_graphs", None)
        if num_graphs is None:
            num_graphs = int(graph0.batch.max().item()) + 1
        u = torch.rand(size=(num_graphs, 1), device=device, dtype=dtype)
        if t_dist_kw == "quadratic":
            u = u ** 2
        elif t_dist_kw != "uniform":
            raise ValueError(f"Unknown t_dist_kw: {t_dist_kw}")
        return u * (self.total_time - t_min) + t_min

    @torch.inference_mode()
    def sample_backward(
        self,
        graphT: Data, # graph carrying the structure (h / edges / num nodes) to denoise into
        score_fn: Callable[[Data, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        # joint score network wrapper: (graph, vt, t) -> (score_l, score_f)
        #   graph    : PyG Batch at the current reverse step (graph.x == f_t, graph.l == l_t)
        #   vt       : (N_total, dim) node velocities
        #   t        : (num_graph, 1) per-graph reverse time
        #   score_l  : (num_graph, 6) lattice score
        #   score_f  : (N_total, dim) fractional-coordinate score (raw TDM score)
        fT_prior_kw: Literal["stdGauss", "uniform"] = "uniform",
        vT_prior_kw: Literal["stdGauss", "uniform"] = "stdGauss",
        lT_prior_kw: Literal["stdGauss", "uniform"] = "stdGauss",
        n_steps: int = 100,
        sample_trajectory: bool = False,
        exponential_integration: bool = True,
        probability_flow: bool = False,
        predictor_corrector: bool = False,
        predictor_corrector_n_steps: int = 1,
        only_correct_vt: bool = False,
        tau: float = 1e-3,
        vT_prior_scale: float = 1.0,
        lT_prior_scale: float = 1.0,
        debug: bool = False,
        progress: bool = False,
        progress_desc: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Joint reverse (denoising) process for the KLDM model.

        Samples the lattice ``l_T`` (via the continuous ``l_diffusion``) and the
        velocity ``v_T`` / fractional coordinates ``f_T`` (via the TDM
        ``f_diffusion``) from their priors, then integrates the coupled reverse
        dynamics step by step.  Each reverse step reuses the per-step update
        primitives already implemented in ``tdm.py`` (velocity/position steps and
        the Langevin corrector) and ``continuous.py`` (Euler backward step and
        the Langevin corrector).

        Returns
        -------
        (l0, f0)                                default
        (l_traj, f_traj, v_traj, t_arr)         sample_trajectory=True
        (l_traj, f_traj, v_traj, t_arr)         debug=True  (same payload)
        """
        assert fT_prior_kw in ["stdGauss", "uniform"]
        assert vT_prior_kw in ["stdGauss", "uniform"]
        assert lT_prior_kw in ["stdGauss", "uniform"]

        f_diff = self.f_diffusion
        l_diff = self.l_diffusion

        graph = graphT.clone()
        device = graph.x.device
        dtype = graph.x.dtype
        batch_vec = graph.batch                                  # (N_total,)
        num_graphs = getattr(graph, "num_graphs", None)
        if num_graphs is None:
            num_graphs = int(batch_vec.max().item()) + 1
        total_nodes = graph.x.shape[0]
        data_dim = graph.x.shape[1]

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        # ------------------------------------------------------------------
        # sample the priors: fractional coordinates f_T, velocities v_T, lattice l_T
        # ------------------------------------------------------------------
        if fT_prior_kw == "stdGauss":
            fT = torch.randn(size=(total_nodes, data_dim), device=device, dtype=dtype, generator=generator)
        else:
            fT = torch.rand(size=(total_nodes, data_dim), device=device, dtype=dtype, generator=generator)
            fT = wrap_angle(pos_to_angle(fT))
        fT = wrap_angle(fT)

        if vT_prior_kw == "stdGauss":
            vT = torch.randn(size=(total_nodes, data_dim), device=device, dtype=dtype, generator=generator) * vT_prior_scale
        else:
            vT = torch.rand(size=(total_nodes, data_dim), device=device, dtype=dtype, generator=generator)
        if f_diff.zero_cog:
            vT = scatter_center(vT, batch_vec)

        if lT_prior_kw == "stdGauss":
            lT = torch.randn(size=(num_graphs, 6), device=device, dtype=dtype, generator=generator) * lT_prior_scale
        else:
            lT = torch.rand(size=(num_graphs, 6), device=device, dtype=dtype, generator=generator)

        # write the priors onto the graph state
        graph = f_diff._update_graph_batch(graph, fT)
        graph = self.update_lattice(graph, lT)

        l_reverse = lT.unsqueeze(-1)                             # (num_graph, 6, 1)
        v_reverse = vT.clone()                                  # (N_total, dim)
        f_reverse = fT.clone()                                  # (N_total, dim)

        # ------------------------------------------------------------------
        # time schedule: total_time -> 0, shared across all graphs at each step
        # ------------------------------------------------------------------
        dt = torch.tensor(self.total_time / (n_steps - 1), device=device, dtype=dtype)
        t_line = torch.linspace(self.total_time, 0, n_steps, device=device, dtype=dtype)
        t_reverse_list = t_line.view(1, n_steps, 1).expand(num_graphs, -1, -1)  # (num_graph, n_steps, 1)

        l_traj, f_traj, v_traj, t_list = [], [], [], []
        if sample_trajectory or debug:
            l_traj.append(l_reverse.squeeze(-1).cpu())
            f_traj.append(f_reverse.cpu())
            v_traj.append(v_reverse.cpu())
        t_list.append(float(t_reverse_list[0, 0, 0].item()))

        step_iter = range(n_steps - 1)
        if progress:
            step_iter = tqdm(step_iter, total=n_steps - 1, desc=progress_desc or "kldm reverse", leave=False)

        for i in step_iter:
            t_g = t_reverse_list[:, i, :]                        # (num_graph, 1)
            t_g_next = t_reverse_list[:, i + 1, :]               # (num_graph, 1)
            t_n = t_g[batch_vec]                                 # (N_total, 1)
            t_l = t_g.unsqueeze(-1)                              # (num_graph, 1, 1) broadcasts over lattice

            # ---- joint score at the current state ----
            score_l, score_f = score_fn(graph, v_reverse, t_g)
            score_l = self._as_lattice_score(score_l)           # (num_graph, 6, 1)

            # ---- velocity predictor (TDM) ----
            scorev = f_diff._construct_scorev(score_f, v_reverse, t_n)
            pf_v = probability_flow or predictor_corrector
            if exponential_integration:
                v_reverse = f_diff.backward_vstep_exp_graph(
                    v_reverse, t_n, batch_vec, dt, scorev, probability_flow=pf_v,
                )
            else:
                v_reverse = f_diff.backward_vstep_euler_graph(
                    v_reverse, t_n, batch_vec, dt, scorev, probability_flow=pf_v,
                )

            # ---- position predictor (TDM) ----
            f_reverse = f_diff.backward_fstep_graph(f_reverse, v_reverse, dt)
            graph = f_diff._update_graph_batch(graph, f_reverse)

            # ---- lattice predictor (continuous Euler backward step) ----
            # under predictor_corrector the predictor is the deterministic ODE,
            # matching the velocity predictor above.
            l_reverse = l_diff.backward_step_euler(
                l_reverse, t_l, score_l, dt, probability_flow=pf_v,
            )
            graph = self.update_lattice(graph, l_reverse.squeeze(-1))

            # ---- predictor-corrector (Langevin) ----
            if predictor_corrector and i < n_steps - 2:
                t_n_next = t_g_next[batch_vec]
                t_l_next = t_g_next.unsqueeze(-1)
                corr_step_size = tau * (t_g_next ** 2) / (self.total_time ** 2)  # (num_graph, 1)
                corr_tau_f = float(corr_step_size.mean().item())
                for _ in range(predictor_corrector_n_steps):
                    score_l, score_f = score_fn(graph, v_reverse, t_g_next)
                    score_l = self._as_lattice_score(score_l)
                    scorev = f_diff._construct_scorev(score_f, v_reverse, t_n_next)
                    if only_correct_vt:
                        v_reverse = f_diff._langevin_corrector_step_graph(
                            graph, v_reverse, t_n_next, batch_vec, dt,
                            scorev, tau=corr_tau_f, only_correct_vt=True,
                        )
                    else:
                        f_reverse, v_reverse = f_diff._langevin_corrector_step_graph(
                            graph, v_reverse, t_n_next, batch_vec, dt,
                            scorev, tau=corr_tau_f, only_correct_vt=False,
                        )
                        graph = f_diff._update_graph_batch(graph, f_reverse)
                        # lattice corrector shares the same annealed step size
                        l_reverse = l_diff.langevin_corrector_step(
                            l_reverse, t_l_next, score_l, dt, tau=corr_tau_f,
                        )
                        graph = self.update_lattice(graph, l_reverse.squeeze(-1))

            t_list.append(float(t_g_next[0, 0].item()))
            if sample_trajectory or debug:
                l_traj.append(l_reverse.squeeze(-1).cpu())
                f_traj.append(f_reverse.cpu())
                v_traj.append(v_reverse.cpu())

        l0 = l_reverse.squeeze(-1)
        f0 = f_reverse

        t_arr = torch.tensor(t_list).numpy()
        if debug or sample_trajectory:
            return l_traj, f_traj, v_traj, t_arr
        return l0, f0

    @staticmethod
    def _as_lattice_score(score_l: torch.Tensor) -> torch.Tensor:
        """Normalize a lattice score to the (num_graph, 6, 1) layout expected by l_diffusion."""
        if score_l.ndim == 2:
            score_l = score_l.unsqueeze(-1)
        return score_l
    

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ):
        raise NotImplementedError

    def update_lattice(self, graph: Data, noised_l: torch.Tensor) -> Data:
        """
        Write a per-graph lattice tensor back onto a PyG batch.

        ``noised_l`` has shape (num_graph, 6): one lattice vector per
        sub-graph. ``l`` is a *graph-level* attribute (one row per graph, not
        one row per node), so assigning it directly keeps the (num_graph, 6)
        layout in the batch. When the batch is later split via
        ``batch.to_data_list()``, each sub-graph receives its own (6,) / (1, 6)
        lattice slice automatically.
        """
        num_graphs = getattr(graph, "num_graphs", None)
        if num_graphs is None:
            num_graphs = int(graph.batch.max().item()) + 1
        assert noised_l.shape[0] == num_graphs, (
            f"expected {num_graphs} lattice rows, got {noised_l.shape[0]}"
        )
        graph.l = noised_l
        return graph

