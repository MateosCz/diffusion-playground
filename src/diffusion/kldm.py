import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence, Callable, Tuple
from src.diffusion.sde import VPSDE, BaseSDE, BaseSDEIntegrator, EulerIntegrator, LinearSchedule
from src.distribution import WrappedNormalDistribution
from src.diffusion.tdm import TDMDiffusion, BaseDiffusion
from src.diffusion.continuous import ContinuousDiffusion
from torch_geometric.data import Data


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
        return_time: bool = True,
        ):
        # initialize the score and time dictionaries
        score_dict = {}
        t_dict = {}

        # sample the lattice vectors
        l = graph0.l # (num_graph, 6)
        l_expanded = l.unsqueeze(-1) # (num_graph, 6, 1)
        noised_l, score_l, t_l = self.l_diffusion.sample_forward(l_expanded, t_min,return_time=True)
        noised_l = noised_l.squeeze(-1) # (num_graph, 6)
        score_dict["l"] = score_l
        t_dict["l"] = t_l

        # sample the fractional coordinates
        # sample_forward_graph returns (vts, noised_graph): the second element
        # is a *cloned* PyG batch whose node coords are already set to f_t.
        (noised_v, noised_graph), score_f, t_f = self.f_diffusion.sample_forward_graph(graph0, t_min=t_min, return_time=True)
        score_dict["f"] = score_f
        t_dict["f"] = t_f

        # construct the noised PyG data: write the noised lattice (num_graph, 6)
        # back onto the batch so every sub-graph carries its own (6,) lattice.
        noised_graph = self.update_lattice(noised_graph, noised_l)

        return (noised_v, noised_graph), score_dict, t_dict

    def sample_backward(
        self,
        graphT: Data, # noised graph at time T
        fT_prior_kw: Literal["stdGauss", "uniform"],
        vT_prior_kw: torch.Tensor,
        n_steps: int = 100,
        sample_trajectory: bool = False,
        exponential_integration: bool = True,
        probability_flow: bool = False,
        predictor_corrector: bool = False,
        predictor_corrector_n_steps: int = 1,
        only_correct_vt: bool = False,
        tau: float = 1e-3,
        
    ):
        raise NotImplementedError
    

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

