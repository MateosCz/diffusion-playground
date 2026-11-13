import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter
from typing import Callable
from src.nn.scoreNNBlock import SinEmbedding, FourierEmbedding
from torch_geometric.data import Data
import numpy as np
from src.nn.ACSPNet import Ambient_CSPLayer
from typing import Literal
from src.dataLib.type_encoding import build_type_encoding
from src.dataLib.type_encoding_state import export_type_encoding, restore_type_encoding

class Ambient_CSPNet_Token(nn.Module):
    """Two-time endpoint GNN for fractional coordinates and lattice parameters.

    graph.x/pos hold ambient cosine/sine pairs, shape (N, f_dim).
    Atom types are represented by tokens or learned embedding or one-hot encoding.
    atom_type_rep: 
        "tokens": use subatomic tokenization from Crystalite(Veljkovic et al., 2026).
        "embedding": Raw Atomic number + learned embedding (nn.Embedding)
        "one-hot": use a one-hot encoding for atom types
    Outputs are unprojected model-space endpoints.
    """

    def __init__(
            self,
            hidden_dim: int = 128,
            time_dim: int = 128, 
            num_layers: int = 4,
            h_dim: int = 100,
            num_freqs: int = 32,
            dis_emb_dim: int = 128,
            ln: bool = True,
            atom_type_rep: Literal["subatomic_tokenizer_pca_16", "subatomic_tokenizer_pca_8", "subatomic_tokenizer_pca_24", "subatomic_tokenizer_raw", "atomic_number", "embedding", "one-hot"] = "subatomic_tokenizer_pca_16",
            pred_h: bool = False,
            pred_l: bool = True,
            time_emb: nn.Module = None,
            f_dim: int = 6, # ambient (cosine/sine) pairs for 3D data 3x2
            radii_emb_dim: int = 128,
            one_hot_dim: int = 100,
            time_mlp_output_dim: int = 128,
            atom_mlp_output_dim: int = 128,
            lattice_mlp_output_dim: int = 128,
            posdiff_mlp_output_dim: int = 128,
            type_encoding_state: dict | None = None,
            initialize_type_encoding: bool = True,

            # neighbor_head: bool = False,
            # composition_context: bool = False,
            # neighbor_hidden_dim: int = 128,
            # context_hidden_dim: int = 128,
    ):
        super(Ambient_CSPNet_Token, self).__init__()
        if f_dim < 2 or f_dim % 2:
            raise ValueError("f_dim must contain complete cosine/sine pairs")
        # if not isinstance(num_freqs, int) or num_freqs < 1:
        #     raise ValueError("num_freqs must be a positive integer")

        self.act_fn = nn.SiLU()
        self.f_dim = f_dim
        # self.num_freqs = num_freqs
        self.atom_type_rep = atom_type_rep
        self.atom_type_dim = None

        # Embedding layers
        if atom_type_rep == "embedding":
            # we just need to embed the given discrete h
            self.node_embedding = nn.Embedding(h_dim + 1, hidden_dim)
            self.atom_type_dim = hidden_dim
        elif atom_type_rep == "subatomic_tokenizer_raw":
            self.atom_type_dim = 34 # use the default token dimension from Crystalite 34 fixed dimesnal subatomic descriptor + PCA16 representation
        elif atom_type_rep == "subatomic_tokenizer_pca_16":
            self.atom_type_dim = 16
        elif atom_type_rep == "subatomic_tokenizer_pca_8":
            self.atom_type_dim = 8
        elif atom_type_rep == "subatomic_tokenizer_pca_24":
            self.atom_type_dim = 24
        elif atom_type_rep == "atomic_number":
            self.atom_type_dim = 100 # use max 100 atomic numbers for the encoding
        elif atom_type_rep == "one-hot":
            self.atom_type_dim = one_hot_dim
        if self.atom_type_dim is None:
            raise ValueError(f"Unsupported atom_type_rep: {atom_type_rep!r}")

        self._type_encoding_state = None
        self._type_encoding_source = "uninitialized"
        if atom_type_rep.startswith("subatomic_tokenizer_"):
            self.register_buffer("atom_type_table", torch.zeros(101, self.atom_type_dim))
            if type_encoding_state is not None:
                self.set_type_encoding_state(type_encoding_state)
            elif initialize_type_encoding:
                self.set_type_encoding_state(export_type_encoding(
                    build_type_encoding(atom_type_rep, vz=100)
                ))
                self._type_encoding_source = "fresh"
        elif type_encoding_state is not None:
            raise ValueError(f"{atom_type_rep} does not use a descriptor encoding state.")
        self.dis_emb_dim = dis_emb_dim
        self.radii_emb_dim = radii_emb_dim
        self.atom_mlp_output_dim = atom_mlp_output_dim
        self.lattice_mlp_output_dim = lattice_mlp_output_dim
        self.posdiff_mlp_output_dim = posdiff_mlp_output_dim
        self.time_mlp_output_dim = time_mlp_output_dim
        self.radii_emb = nn.Sequential(
            nn.Linear(f_dim // 2, self.radii_emb_dim),
            nn.SiLU(),
            nn.Linear(self.radii_emb_dim, self.radii_emb_dim),
        )

        
        self.pos_emb = SinEmbedding(n_frequencies=num_freqs, n_space=f_dim // 2)
        self.dis_emb = nn.Sequential(
            nn.Linear(self.posdiff_mlp_output_dim, self.dis_emb_dim),
            nn.SiLU(),
            nn.Linear(self.dis_emb_dim, self.dis_emb_dim),
        )

        if time_emb is None:
            time_emb = FourierEmbedding(in_features=1, out_features=time_dim)

        self.time_emb = time_emb

        self.time_mlp = nn.Sequential(
            nn.Linear(2*time_dim, time_mlp_output_dim),
            nn.ReLU(),
            nn.Linear(time_mlp_output_dim, time_mlp_output_dim),
            nn.ReLU(),
            )
        self.atom_type_emb_mlp = nn.Sequential(
            nn.Linear(self.atom_type_dim, self.atom_mlp_output_dim),
            nn.SiLU(),
            nn.Linear(self.atom_mlp_output_dim, self.atom_mlp_output_dim),
        )
        self.lattice_emb_mlp = nn.Sequential(
            nn.Linear(6, self.lattice_mlp_output_dim),
            # nn.SiLU(),
            # nn.Linear(self.lattice_mlp_output_dim, self.lattice_mlp_output_dim),
        )
        self.posdiff_emb_mlp = nn.Sequential(
            nn.Linear(self.f_dim, self.posdiff_mlp_output_dim),
            nn.SiLU(),
            nn.Linear(self.posdiff_mlp_output_dim, self.posdiff_mlp_output_dim),
        )
        
        self.atom_latent_emb = nn.Linear(self.atom_mlp_output_dim + self.time_mlp_output_dim + self.radii_emb_dim, hidden_dim)


        # Message-passing layers
        self.layers = nn.ModuleList(
            [
                Ambient_CSPLayer(
                    dis_emb=self.dis_emb,
                    dis_emb_dim=self.dis_emb_dim,
                    hidden_dim=hidden_dim,
                    l_dim=self.lattice_mlp_output_dim,
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
        
        self.pred_h = pred_h
        self.pred_l = pred_l



        # Construct optional modules last so legacy seeded initialization and
        # checkpoint keys remain unchanged when both additions are disabled.
        # self.neighbor_head = neighbor_head
        # self.composition_context = composition_context
        # if neighbor_head:
        #     self.neighbor_coeff_mlp = nn.Sequential(
        #         nn.Linear(hidden_dim * 2 + f_dim + 6, neighbor_hidden_dim),
        #         nn.SiLU(),
        #         nn.Linear(neighbor_hidden_dim, f_dim),
        #     )
        #     nn.init.zeros_(self.neighbor_coeff_mlp[-1].weight)
        #     nn.init.zeros_(self.neighbor_coeff_mlp[-1].bias)
        # if composition_context:
        #     self.composition_context_mlp = nn.Sequential(
        #         nn.Linear(hidden_dim + 1, context_hidden_dim),
        #         nn.SiLU(),
        #         nn.Linear(context_hidden_dim, hidden_dim),
        #     )
        #     nn.init.zeros_(self.composition_context_mlp[-1].weight)
        #     nn.init.zeros_(self.composition_context_mlp[-1].bias)

    def set_type_encoding_state(self, state: dict) -> None:
        """Restore the fitted element table without recomputing descriptors/PCA."""
        encoding = restore_type_encoding(state)
        if encoding.name != self.atom_type_rep or encoding.type_dim != self.atom_type_dim:
            raise ValueError(
                f"Encoding {encoding.name} ({encoding.type_dim} features) does not match "
                f"model {self.atom_type_rep} ({self.atom_type_dim} features)."
            )
        if not hasattr(self, "atom_type_table"):
            raise ValueError(f"{self.atom_type_rep} does not use a descriptor encoding state.")
        self.atom_type_table = encoding.z_to_enc.to(device=self.atom_type_table.device).clone()
        self._type_encoding_state = export_type_encoding(encoding)
        self._type_encoding_source = "supplied"

    def get_type_encoding_state(self) -> dict | None:
        if self._type_encoding_state is None:
            return None
        state = export_type_encoding(restore_type_encoding(self._type_encoding_state))
        if not torch.equal(self.atom_type_table.detach().cpu(), state["encoding_state"]["z_to_enc"]):
            raise ValueError("Model element lookup table disagrees with its saved encoding state.")
        return state

    def encode_atom_types(self, h: torch.Tensor) -> torch.Tensor:
        """Encode atomic numbers online; cached graph.avec is deliberately unused."""
        if h.dtype != torch.long or h.ndim != 1:
            raise ValueError("graph.h must be a one-dimensional LongTensor of atomic numbers.")
        if self.atom_type_rep == "embedding":
            return self.node_embedding(h)
        if self.atom_type_rep == "atomic_number":
            return F.one_hot(h - 1, num_classes=self.atom_type_dim).to(dtype=self.atom_latent_emb.weight.dtype)
        if self.atom_type_rep == "one-hot":
            return F.one_hot(h, num_classes=self.atom_type_dim).to(dtype=self.atom_latent_emb.weight.dtype)
        if self._type_encoding_state is None:
            raise RuntimeError(
                "Model has no fitted atom encoding. Restore a checkpoint with "
                "type_encoding_state or supply the original training encoding."
            )
        if h.numel() and (h.min() < 1 or h.max() >= self.atom_type_table.shape[0]):
            raise ValueError(f"Atomic numbers must be in [1, {self.atom_type_table.shape[0] - 1}].")
        return self.atom_type_table[h]

    def forward(
            self,
            graph: Data,
            s: torch.Tensor,
            t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pos = graph.pos
        h = graph.h
        l = graph.l
        l_raw = l
        batch = graph.batch
        edge_index = graph.edge_index
        if pos.shape[-1] != self.f_dim:
            raise ValueError(f"graph.pos must have {self.f_dim} ambient features")

        # Use Categorical Flow Maps time embedding style
        t = t-s
        s = self.time_emb(s)
        t = self.time_emb(t)
        time_per_atom = torch.cat([s, t], dim=-1)[batch]
        time_per_atom = self.time_mlp(time_per_atom)

        pairs = pos.reshape(-1, self.f_dim // 2, 2)
        radii_squared = pairs.square().sum(-1)
        radii_features = self.radii_emb(radii_squared)
        node_features = self.encode_atom_types(h)

        # if self.composition_context:
        #     composition = scatter(
        #         node_features, batch, dim=0, reduce="mean", dim_size=l.shape[0],
        #     )
        #     counts = scatter(
        #         node_features.new_ones((node_features.shape[0], 1)),
        #         batch, dim=0, reduce="sum", dim_size=l.shape[0],
        #     )
        #     context = self.composition_context_mlp(
        #         torch.cat([composition, counts.log()], dim=-1),
        #     )
        node_features = self.atom_type_emb_mlp(node_features)
        l = self.lattice_emb_mlp(l)
        node_features = torch.cat([node_features, time_per_atom, radii_features], dim=1) 
        node_features = self.atom_latent_emb(node_features)
        # if self.composition_context:
        #     node_features = node_features + context[batch]

        # Relative dot/cross products are invariant under periodic translations.
        normalized = pairs / (radii_squared + 1e-8).sqrt().unsqueeze(-1) 
        source, target = normalized[edge_index[0]], normalized[edge_index[1]] # (E,3,2) R^2xR^2xR^2, each dim in R^2, total 3 dims
        dot = (source * target).sum(-1) # (E,3) 3 dims dot product
        cross = source[..., 0] * target[..., 1] - source[..., 1] * target[..., 0] # (E,3) cross product
        pos_diff = torch.stack([dot, cross], dim=-1).flatten(start_dim=1) # (E,3,2)->(E,6) rotate invariant dot/cross representation

        # pos_diff_features = self.pos_emb(torch.atan2(cross, dot)) # (E, num_freqs * f_dim) f_dim is ambient features dim (3*2=6)
        # pos_diff_features = self.posdiff_emb_mlp(pos_diff_features)
        pos_diff_features = self.posdiff_emb_mlp(pos_diff)
        edge_graph_index = batch[edge_index[0]]

        for layer in self.layers:
            node_features = layer.forward(
                pos_diff=pos_diff_features,
                # pos_diff=pos_diff,
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
        # if self.neighbor_head:
        #     # Neighbor pairs provide an equivariant direction even when a
        #     # source pair is zero. Self edges do not count as neighbors.
        #     nonself = edge_index[0] != edge_index[1]
        #     source_index, target_index = edge_index[:, nonself]
        #     edge_coefficients = self.neighbor_coeff_mlp(torch.cat([
        #         node_features[source_index], node_features[target_index],
        #         pos_diff[nonself], l[edge_graph_index[nonself]],
        #     ], dim=-1)).reshape(-1, self.f_dim // 2, 2)
        #     neighbor_residual = (
        #         edge_coefficients[..., :1] * pairs[target_index]
        #         + edge_coefficients[..., 1:] * perpendicular_pairs[target_index]
        #     )
        #     residual = residual + scatter(
        #         neighbor_residual, source_index, dim=0, reduce="mean",
        #         dim_size=pairs.shape[0],
        #     )
        out = {"f": pos + residual.flatten(start_dim=1)}

        if self.pred_l:
            graph_features = scatter(node_features, batch, dim=0, reduce="mean", dim_size=l.shape[0])
            # if self.composition_context:
            #     graph_features = graph_features + context
            out_l = self.out_l(graph_features)
            out_l = out_l.view(-1, 6)
            out["l"] = out_l + l_raw

        if self.pred_h:
            out_h = self.out_h(node_features)
            out["h"] = out_h + h

        return out
