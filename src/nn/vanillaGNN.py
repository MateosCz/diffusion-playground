"""
Vanilla GNN score network for TDM on graph-structured data (e.g. shape dataset).

Mirrors the design of TDM_SimpleScoreMLP:
  - Sinusoidal time embedding → lifting MLP → inject at every layer
  - Separate lifting for spatial (node) and time features
  - LayerNorm (no BatchNorm)
  - SiLU activations throughout

Input:
  A PyG Data batch from PyGGraphWrapper containing:
    data.x          : (N_total, node_feat_dim)  -- node features (angles + optional Fourier)
    data.pos        : (N_total, 2)              -- raw fractional coordinates
    data.edge_index : (2, E_total)              -- fully-connected edges
    data.edge_attr  : (E_total, edge_feat_dim)  -- wrapped angular diffs + sin/cos
    data.batch      : (N_total,)                -- batch assignment

  vt : (N_total, 2)  -- velocity at each node
  t  : (B, 1)        -- diffusion time per graph in the batch

Output:
  score : (N_total, output_dim) -- predicted score at each node
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from typing import Sequence, Union

import src.nn.scoreNNBlock as Block


# ---------------------------------------------------------------------------
# Message-passing layer with time conditioning
# ---------------------------------------------------------------------------
class TimeConditionedMPLayer(MessagePassing):
    """
    A single message-passing layer that:
      1. Computes edge messages from (h_i, h_j, edge_attr) via an edge MLP
      2. Aggregates messages (sum)
      3. Updates node state: h_i' = MLP([h_i, agg_msg, h_t])

    Time is injected at the update step via concatenation, matching the MLP's
    pattern of concatenating h_t at each hidden layer.
    """

    def __init__(self, node_dim: int, edge_feat_dim: int, time_dim: int):
        super().__init__(aggr="sum")

        # Edge message MLP: (h_i || h_j || edge_attr) -> message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_feat_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # Node update MLP: (h_i || aggregated_msg || h_t) -> h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + time_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        h          : (N_total, node_dim)
        edge_index : (2, E_total)
        edge_attr  : (E_total, edge_feat_dim)
        h_t        : (N_total, time_dim)  -- already expanded per node
        """
        # propagate calls message() then aggregate()
        agg = self.propagate(edge_index, h=h, edge_attr=edge_attr)

        # update with residual
        h_updated = self.node_mlp(torch.cat([h, agg, h_t], dim=-1))
        h = self.norm(h + h_updated)  # residual + LayerNorm
        return h

    def message(self, h_i: torch.Tensor, h_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1))


# ---------------------------------------------------------------------------
# Main GNN score network
# ---------------------------------------------------------------------------
class TDM_VanillaGNN(nn.Module):
    """
    Graph-based score network for TDM, analogous to TDM_SimpleScoreMLP.

    Parameters
    ----------
    node_feat_dim : int
        Dimension of input node features from PyGGraphWrapper (e.g. 6 with Fourier).
    edge_feat_dim : int
        Dimension of edge attributes from PyGGraphWrapper (e.g. 6).
    v_dim : int
        Dimension of velocity per node (typically 2 for T^2).
    hidden_dim : int or Sequence[int]
        Width(s) of the message-passing layers. If int, uses that width for
        all `num_mp_layers` layers.
    num_mp_layers : int
        Number of message-passing layers (used only if hidden_dim is int).
    time_embedding_half_dim : int
        Half-dimension for sinusoidal time embedding (full dim = 2x).
    output_dim : int
        Output dimension per node (score dimension).
    total_time : float
        Total diffusion time (for normalizing t before embedding).
    time_embedding_scale : float
        Scale factor applied to t/T before sinusoidal embedding.
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        edge_feat_dim: int = 6,
        v_dim: int = 2,
        hidden_dim: Union[int, Sequence[int]] = 128,
        num_mp_layers: int = 4,
        time_embedding_half_dim: int = 64,
        output_dim: int = 2,
        total_time: float = 2.0,
        time_embedding_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.v_dim = v_dim
        self.output_dim = output_dim
        self.total_time = total_time
        self.time_embedding_scale = time_embedding_scale
        self.time_embedding_half_dim = time_embedding_half_dim
        self.time_embedding_dim = 2 * time_embedding_half_dim

        # Build list of hidden widths
        if isinstance(hidden_dim, int):
            self.hidden_dims = [hidden_dim] * num_mp_layers
        else:
            self.hidden_dims = list(hidden_dim)

        node_input_dim = node_feat_dim + v_dim  # concatenate node features + velocity

        # --- Lifting layers (same pattern as MLP) ---
        self.lifting_layer_x = nn.Sequential(
            nn.Linear(node_input_dim, self.hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[0]),
        )

        self.lifting_layer_t = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        self.input_norm = nn.LayerNorm(self.hidden_dims[0])

        # --- Message-passing layers ---
        self.mp_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            self.mp_layers.append(
                TimeConditionedMPLayer(
                    node_dim=self.hidden_dims[i],
                    edge_feat_dim=edge_feat_dim,
                    time_dim=self.time_embedding_dim,
                )
            )

        # --- Output projection ---
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(self.hidden_dims[-1], output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        vt: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N_total, node_feat_dim)  node features from PyGGraphWrapper
        vt         : (N_total, v_dim)          velocity at each node
        t          : (B, 1)                    diffusion time per graph
        edge_index : (2, E_total)              edge connectivity
        edge_attr  : (E_total, edge_feat_dim)  edge features
        batch      : (N_total,)                graph membership

        Returns
        -------
        score : (N_total, output_dim)
        """
        # --- Time embedding (per-graph, then expand to per-node) ---
        t_norm = t / self.total_time                          # (B, 1)
        t_emb = Block.sinusoidal_time_embedding(
            t_norm.squeeze(-1) * self.time_embedding_scale,
            self.time_embedding_half_dim,
        )                                                     # (B, time_emb_dim)
        h_t_graph = self.lifting_layer_t(t_emb)               # (B, time_emb_dim)
        h_t = h_t_graph[batch]                                # (N_total, time_emb_dim)

        # --- Node lifting ---
        h = self.lifting_layer_x(torch.cat([x, vt], dim=-1))  # (N_total, hidden_dims[0])
        h = self.input_norm(h)

        # --- Message passing ---
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_attr, h_t)

        # --- Output ---
        score = self.output_layer(h)                           # (N_total, output_dim)
        return score

    def forward_from_data(self, data, vt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convenience method that unpacks a PyG Data/Batch object.

        data : torch_geometric.data.Data or Batch with .x, .edge_index, .edge_attr, .batch
        vt   : (N_total, v_dim)
        t    : (B, 1)
        """
        return self.forward(
            x=data.x,
            vt=vt,
            t=t,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )