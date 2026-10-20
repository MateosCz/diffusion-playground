import torch
from torch import nn
from torch_geometric.utils import scatter

from src.nn.scoreNNBlock import SinEmbedding, FourierEmbedding
from torch_geometric.data import Data


class Ambient_CSPLayer(nn.Module):
    def __init__(
            self,
            dis_emb: nn.Module,
            dis_emb_dim: int,
            hidden_dim: int = 128,
            act_fn: nn.Module = nn.SiLU(),
            ln: bool = False,
    ):
        super(Ambient_CSPLayer, self).__init__()

        self.dis_emb = dis_emb
        self.dis_emb_dim = dis_emb_dim

        input_dim = (
                hidden_dim * 2 + self.dis_emb_dim + 6
        )  # hidden states + relative geometry + lattice

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
            self,
            pos_diff: torch.Tensor,
            node_features: torch.Tensor,
            lattices: torch.Tensor,
            edge_index: torch.Tensor,
            edge_graph_index: torch.Tensor,
    ):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]

        pos_diff = self.dis_emb(pos_diff)
        l_edge = lattices[edge_graph_index]

        edges_input = torch.cat([hi, hj, l_edge, pos_diff], dim=1)

        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(
            self,
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            edge_index: torch.Tensor,
    ):
        agg = scatter(
            edge_features,
            edge_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
            self,
            pos_diff: torch.Tensor,
            node_features: torch.Tensor,
            l: torch.Tensor,
            edge_index: torch.Tensor,
            edge_graph_index: torch.Tensor,
    ):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)

        edge_features = self.edge_model(
            pos_diff=pos_diff,
            node_features=node_features,
            lattices=l,
            edge_index=edge_index,
            edge_graph_index=edge_graph_index,
        )

        node_output = self.node_model(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )
        return node_input + node_output


class Ambient_CSPNet(nn.Module):
    """Two-time endpoint GNN for fractional coordinates and lattice parameters.

    graph.x/pos hold ambient cosine/sine pairs, shape (N, f_dim).
    Atom types stay fixed. Outputs are unprojected model-space endpoints.
    """

    def __init__(
            self,
            hidden_dim: int = 128,
            time_dim: int = 128,
            num_layers: int = 4,
            h_dim: int = 100,
            # num_freqs: int = 10,
            dis_emb_dim: int = 128,
            ln: bool = True,
            smooth: bool = False,
            pred_h: bool = False,
            pred_l: bool = True,
            time_emb: nn.Module = None,
            f_dim: int = 6, # ambient (cosine/sine) pairs for 3D data 3x2
            radii_emb_dim: int = 128,
    ):
        super(Ambient_CSPNet, self).__init__()
        if f_dim < 2 or f_dim % 2:
            raise ValueError("f_dim must contain complete cosine/sine pairs")

        self.act_fn = nn.SiLU()
        self.f_dim = f_dim

        # Embedding layers
        if smooth:
            self.node_embedding = nn.Linear(h_dim, hidden_dim, bias=False)
        else:
            # we just need to embed the given discrete h
            self.node_embedding = nn.Embedding(h_dim + 1, hidden_dim)
        self.radii_emb_dim = radii_emb_dim
        self.radii_emb = nn.Sequential(
            nn.Linear(f_dim // 2, self.radii_emb_dim),
            nn.SiLU(),
        )


        self.atom_latent_emb = nn.Linear(hidden_dim + 2 * time_dim + self.radii_emb_dim + 6, hidden_dim)

        # self.dis_emb = SinEmbedding(n_frequencies=num_freqs, n_space=f_dim)
        self.dis_emb_dim = dis_emb_dim
        # self.dis_emb = SinEmbedding(n_frequencies=num_freqs, n_space=f_dim)
        self.dis_emb = nn.Sequential(
            nn.Linear(f_dim, self.dis_emb_dim),
            nn.SiLU(),
        )

        if time_emb is None:
            time_emb = FourierEmbedding(in_features=1, out_features=time_dim)

        self.time_emb = time_emb

        # Message-passing layers
        self.layers = nn.ModuleList(
            [
                Ambient_CSPLayer(
                    dis_emb=self.dis_emb,
                    dis_emb_dim=self.dis_emb_dim,
                    hidden_dim=hidden_dim,
                    act_fn=self.act_fn,
                    ln=ln,
                )
                for _ in range(num_layers)
            ]
        )

        # Readout layers
        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.out_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, f_dim, bias=True),
        )

        if pred_l:
            self.out_l = nn.Linear(hidden_dim, 6, bias=True)

        if pred_h:
            self.out_h = nn.Linear(hidden_dim, h_dim)

        self.ln = ln
        self.smooth = smooth
        self.pred_h = pred_h
        self.pred_l = pred_l

    def forward(
            self,
            graph: Data,
            s: torch.Tensor,
            t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pos = graph.pos
        h = graph.h
        l = graph.l
        batch = graph.batch
        edge_index = graph.edge_index
        if pos.shape[-1] != self.f_dim:
            raise ValueError(f"graph.pos must have {self.f_dim} ambient features")

        s = self.time_emb(s)
        t = self.time_emb(t)
        time_per_atom = torch.cat([s, t], dim=-1)[batch]

        pairs = pos.reshape(-1, self.f_dim // 2, 2)
        radii_squared = pairs.square().sum(-1)
        radii_features = self.radii_emb(radii_squared)

        node_features = self.node_embedding(h)
        node_features = torch.cat([node_features, time_per_atom, radii_features, l[batch]], dim=1) 
        node_features = self.atom_latent_emb(node_features)

        # Relative dot/cross products are invariant under periodic translations.
        # Smooth normalization also keeps the time JVP finite at zero states.
        normalized = pairs / (radii_squared + 1e-8).sqrt().unsqueeze(-1) 
        source, target = normalized[edge_index[0]], normalized[edge_index[1]] # (E,3,2) R^2xR^2xR^2, each dim in R^2, total 3 dims
        dot = (source * target).sum(-1) # (E,3) 3 dims dot product
        cross = source[..., 0] * target[..., 1] - source[..., 1] * target[..., 0] # (E,3) cross product
        pos_diff = torch.stack([dot, cross], dim=-1).flatten(start_dim=1) # (E,3,2) rotate invariant dot/cross representation
        edge_graph_index = batch[edge_index[0]]

        for layer in self.layers:
            node_features = layer.forward(
                pos_diff=pos_diff,
                node_features=node_features,
                l=l,
                edge_index=edge_index,
                edge_graph_index=edge_graph_index,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # Express the coordinate residual in the local pair/J(pair) basis.
        # Rotating the source pairs therefore rotates the predicted endpoint.
        coefficients = self.out_f(node_features).reshape_as(pairs)
        perpendicular_pairs = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1) # counterclockwise rotate vectors by 90 degrees, so it is the orthogonal vector
        residual = coefficients[..., :1] * pairs + coefficients[..., 1:] * perpendicular_pairs # first term is radial component, second term is tangential component
        out = {"f": pos + residual.flatten(start_dim=1)}

        if self.pred_l:
            graph_features = scatter(node_features, batch, dim=0, reduce="mean", dim_size=l.shape[0])
            out_l = self.out_l(graph_features)
            out_l = out_l.view(-1, 6)
            out["l"] = out_l + l

        if self.pred_h:
            out_h = self.out_h(node_features)
            out["h"] = out_h + h

        return out
