"""Joint CSP flow-map training, with one shared time per crystal."""

import math

import lightning as L
import torch
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn as nn

from src.nn.ACSPNet import Weight_MLP


class LitCSPFlowMap(L.LightningModule):
    def __init__(
        self, pi, flow_map, eta=0.5, lr=1e-3, lambda_f=1.0, lambda_l=1.0,
        beta_f=1.0, beta_l=1.0,
        warmup_steps: int = 1000,
        with_ema: bool = True,
        ema_decay:float = 0.999,
        ema_start: int = 100,
        weight_mlp_kwargs: dict | None = None,
        use_weight_mlp: bool = True,
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
        self.use_weight_mlp = use_weight_mlp
        nn.init.normal_(
                self.pi.out_f[-1].weight,
                mean=0.0,
                std=0.01,
            )
        nn.init.zeros_(self.pi.out_f[-1].bias)

        nn.init.normal_(self.pi.out_l.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.pi.out_l.bias)

        weight_mlp_kwargs = {
            "n_hidden": 1, "n_neurons": pi.out_l.in_features,
            "use_residual": False, **(weight_mlp_kwargs or {}),
        }
        self.weight_mlp = Weight_MLP(
            input_dim=2, output_dim=2, act=nn.GELU(), **weight_mlp_kwargs,
        )
        nn.init.zeros_(self.weight_mlp.output_layer.weight)
        nn.init.zeros_(self.weight_mlp.output_layer.bias)

        self.warmup_steps = warmup_steps
        if with_ema:
            self.ema_pi = torch.optim.swa_utils.AveragedModel(
                self.pi,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            )
            self.ema_start = ema_start
            self.ema_pi.requires_grad_(False)

        else:
            self.ema_pi = None

        self.save_hyperparameters(ignore=["pi", "flow_map", "rg_vfm", "l_vfm"])

        

    def forward(self, graph, s, t):
        """Return the coordinate and lattice endpoints under keys f/l."""
        return self.pi(graph, s, t)

    def weighted_loss(self, losses, s, t):
        """Flow Maps log weights, one column each for coordinates and lattice."""
        if not self.use_weight_mlp:
            return {key: losses[key].mean() for key in ("f", "l")}
        weight = self.weight_mlp(torch.cat([s, t], dim=-1))
        return {
            key: (torch.exp(-weight[:, i]) * losses[key] + weight[:, i]).mean()
            for i, key in enumerate(("f", "l"))
        }

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
        if stage == "train":
            self._loss_spike_inputs = {"diagonal": (target, graph_t, t)}
        tangent = self.flow_map.tangent_loss(target, graph_t, t, self.pi)
        weighted_tangent = self.weighted_loss(tangent, t, t)

        # Uniform over the time triangle, with the state constructed at s.
        a = self.rg_vfm.sample_time(off_diagonal.l)
        b = self.rg_vfm.sample_time(off_diagonal.l)
        s, t = torch.minimum(a, b), torch.maximum(a, b)
        _, graph_s, _ = self.sample_training_pair(off_diagonal, t=s)
        if stage == "train":
            self._loss_spike_inputs["off_diagonal"] = (graph_s, s, t)
        lagrange = self.flow_map.lagrange_loss(graph_s, s, t, self.pi)
        weighted_lagrange = self.weighted_loss(lagrange, s, t)
        loss = (
            self.lambda_f * (weighted_tangent["f"] + self.beta_f * weighted_lagrange["f"])
            + self.lambda_l * (weighted_tangent["l"] + self.beta_l * weighted_lagrange["l"])
        )
        self.log_dict(
            {
                f"{stage}/total_loss": loss,
                f"{stage}/tangent_loss_f": tangent["f"].mean(),
                f"{stage}/tangent_loss_l": tangent["l"].mean(),
                f"{stage}/lagrange_loss_f": lagrange["f"].mean(),
                f"{stage}/lagrange_loss_l": lagrange["l"].mean(),
                f"{stage}/tangent_contribution_f": self.lambda_f * weighted_tangent["f"],
                f"{stage}/tangent_contribution_l": self.lambda_l * weighted_tangent["l"],
                f"{stage}/lagrange_contribution_f": self.lambda_f * self.beta_f * weighted_lagrange["f"],
                f"{stage}/lagrange_contribution_l": self.lambda_l * self.beta_l * weighted_lagrange["l"],
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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (
                self.ema_pi is not None
                and self.trainer.current_epoch > self.hparams.ema_start
        ):
            self.ema_pi.update_parameters(self.pi)

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
        if self.ema_pi is not None and self.ema_pi.n_averaged.item() > 0:
            pi = self.ema_pi.module
        else:
            pi = self.pi
        return self.flow_map.sample(graph0, pi, n_steps=n_steps)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [*self.pi.parameters(), *self.weight_mlp.parameters()],
            lr=self.lr, amsgrad=True, foreach=True, weight_decay=1e-2,eps=1e-8,
        )

        min_lr_ratio = 0.01
        total_steps = int(self.trainer.estimated_stepping_batches)

        assert self.warmup_steps < total_steps

        scheduler = SequentialLR(opt, [
            LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - self.warmup_steps, eta_min=min_lr_ratio * self.lr),
        ], milestones=[self.warmup_steps])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

