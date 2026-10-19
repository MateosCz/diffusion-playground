"""Joint CSP flow-map training, with one shared time per crystal."""

import math

import lightning as L
import torch
from torch_geometric.data import Batch



class LitCSPFlowMap(L.LightningModule):
    def __init__(
        self, pi, flow_map, eta=0.5, lr=1e-3, lambda_f=1.0, lambda_l=1.0,
        beta_f=1.0, beta_l=1.0,
    ):
        super().__init__()
        if not 0 < eta <= 1:
            raise ValueError("eta must be in (0, 1]")
        if not all(math.isfinite(value) and value >= 0 for value in (beta_f, beta_l)):
            raise ValueError("Lagrange weights beta_f/beta_l must be finite and non-negative")
        self.pi = pi
        self.flow_map = flow_map
        self.rg_vfm = flow_map.f_flowmap.vfm
        self.l_vfm = flow_map.l_flowmap.vfm
        self.eta = eta
        self.lr = lr
        self.lambda_f = lambda_f
        self.lambda_l = lambda_l
        self.beta_f = beta_f
        self.beta_l = beta_l

    def forward(self, graph, s, t):
        """Return the coordinate and lattice endpoints under keys f/l."""
        return self.pi(graph, s, t)

    def sample_training_pair(self, graph1, graph0=None, *, t=None):
        """Return (t, graph_t, target_graph), with times of shape (B, 1)."""
        t = self.rg_vfm.sample_time(graph1.l, t=t)
        _, f_t, f1 = self.rg_vfm.sample_training_pair(
            graph1.x, None if graph0 is None else graph0.x,
            t=t[graph1.batch],
        )
        _, l_t, l1 = self.l_vfm.sample_training_pair(
            graph1.l, None if graph0 is None else graph0.l, t=t,
        )
        return t, self.flow_map._update_graph(graph1, f_t, l_t), self.flow_map._update_graph(graph1, f1, l1)

    @torch.inference_mode(False)
    def _shared_step(self, batch, stage):
        """Split crystals into diagonal tangent and off-diagonal Lagrange pairs."""
        if batch.num_graphs < 2:
            raise ValueError("batch must contain at least 2 crystals")
        n_diagonal = min(max(int(self.eta * batch.num_graphs), 1), batch.num_graphs - 1)
        graphs = batch.to_data_list()
        # PyG may recreate batch/ptr on CPU when collating MPS tensors.
        diagonal = Batch.from_data_list(graphs[:n_diagonal]).to(batch.x.device)
        off_diagonal = Batch.from_data_list(graphs[n_diagonal:]).to(batch.x.device)

        t, graph_t, target = self.sample_training_pair(diagonal)
        tangent = self.flow_map.tangent_loss(target, graph_t, t, self.pi)
        s, graph_s, _ = self.sample_training_pair(off_diagonal)
        t = s + torch.rand_like(s) * (1 - s)
        lagrange = self.flow_map.lagrange_loss(graph_s, s, t, self.pi)
        loss = (
            self.lambda_f * (tangent["f"] + self.beta_f * lagrange["f"])
            + self.lambda_l * (tangent["l"] + self.beta_l * lagrange["l"])
        )
        self.log_dict(
            {
                f"{stage}/total_loss": loss,
                f"{stage}/tangent_loss_f": tangent["f"],
                f"{stage}/tangent_loss_l": tangent["l"],
                f"{stage}/lagrange_loss_f": lagrange["f"],
                f"{stage}/lagrange_loss_l": lagrange["l"],
            },
            on_step=stage == "train", on_epoch=True,
            prog_bar=stage in ("train", "val"), batch_size=batch.num_graphs,
        )
        return loss

    def training_step(self, batch, batch_idx=0):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx=0):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx=0):
        return self._shared_step(batch, "test")

    @torch.inference_mode()
    def sample(self, batch, n_steps=1):
        """Draw priors using the crystal template, then sample from time 0 to 1."""
        f0 = self.rg_vfm.sample_prior(
            (batch.num_nodes, self.rg_vfm.model_dim),
            device=batch.x.device, dtype=batch.x.dtype,
        )
        l0 = self.l_vfm.sample_prior(
            (batch.num_graphs, self.l_vfm.model_dim),
            device=f0.device, dtype=f0.dtype,
        )
        graph0 = self.flow_map._update_graph(batch, f0, l0)
        return self.flow_map.sample(graph0, self.pi, n_steps=n_steps)

    def configure_optimizers(self):
        return torch.optim.Adam(self.pi.parameters(), lr=self.lr)
