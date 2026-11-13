"""Joint CSP flow-map training, with one shared time per crystal."""

import math

import lightning as L
import torch
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn as nn

from src.nn.ACSPNet import Weight_MLP, Weight_MLP_SiLU


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
        non_linear_weight_mlp: bool = False,
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
        self.non_linear_weight_mlp = non_linear_weight_mlp
        if self.use_weight_mlp:
            if self.non_linear_weight_mlp:
                self.weight_mlp_f = Weight_MLP_SiLU(
                    output_dim=1, **(weight_mlp_kwargs or {}), # 2 for coordinate and lattice
                )
                self.weight_mlp_l = Weight_MLP_SiLU(
                    output_dim=1, **(weight_mlp_kwargs or {}), # 2 for coordinate and lattice
                )
            else:
                self.weight_mlp_f = Weight_MLP(
                    output_dim=1, **(weight_mlp_kwargs or {}), # 2 for coordinate and lattice
                )
                self.weight_mlp_l = Weight_MLP(
                    output_dim=1, **(weight_mlp_kwargs or {}), # 2 for coordinate and lattice
                )
        else:
            self.weight_mlp_f = None
            self.weight_mlp_l = None

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

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self.pi, "get_type_encoding_state"):
            state = self.pi.get_type_encoding_state()
            if hasattr(self.pi, "atom_type_table") and state is None:
                raise RuntimeError("Cannot save a model without its fitted atom encoding.")
            if state is not None:
                if self.ema_pi is not None:
                    ema_state = self.ema_pi.module.get_type_encoding_state()
                    if ema_state is None or ema_state["state_sha256"] != state["state_sha256"]:
                        raise ValueError("Predictor and EMA must use the same fitted atom encoding.")
                checkpoint["type_encoding_state"] = state

    def _restore_type_encoding(self, checkpoint):
        if not hasattr(self.pi, "atom_type_table"):
            return
        embedded = checkpoint.get("type_encoding_state")
        supplied = (
            self.pi.get_type_encoding_state()
            if self.pi._type_encoding_source == "supplied" else None
        )
        if embedded is not None and supplied is not None:
            if embedded.get("state_sha256") != supplied["state_sha256"]:
                raise ValueError("Explicit atom encoding conflicts with the checkpoint encoding.")
        state = embedded if embedded is not None else supplied
        if state is None:
            raise RuntimeError(
                "This legacy checkpoint has no type_encoding_state. Supply the original "
                "training file with --type-encoding-state, or migrate it with "
                "python -m src.litTrain.migrateCSPEncoding. "
                "Do not refit PCA or substitute another machine's encoding."
            )
        networks = [("pi.", self.pi)]
        if self.ema_pi is not None:
            networks.append(("ema_pi.module.", self.ema_pi.module))
        for prefix, network in networks:
            network.set_type_encoding_state(state)
            network._type_encoding_source = "checkpoint"
            key = prefix + "atom_type_table"
            table = network.atom_type_table.detach().cpu()
            saved_table = checkpoint["state_dict"].get(key)
            if saved_table is not None:
                if not torch.equal(saved_table.detach().cpu(), table):
                    raise ValueError(f"Checkpoint {key} disagrees with its atom encoding state.")
            elif embedded is not None:
                raise ValueError(f"Checkpoint is missing its saved {key} buffer.")
            else:
                # Old checkpoints acquire only the fixed table; learned weights stay intact.
                checkpoint["state_dict"][key] = table.clone()

    def on_load_checkpoint(self, checkpoint):
        """Migrate a shared two-output loss-weight network to independent f/l nets."""
        self._restore_type_encoding(checkpoint)
        state_dict = checkpoint["state_dict"]
        prefixes = ("weight_mlp.", "weight_mlp_f.", "weight_mlp_l.")
        if not self.use_weight_mlp:
            for key in list(state_dict):
                if key.startswith(prefixes):
                    del state_dict[key]
            return

        legacy_prefix = "weight_mlp."
        legacy_keys = [key for key in state_dict if key.startswith(legacy_prefix)]
        if not legacy_keys:
            return  # Current two-network checkpoints already have the right keys.
        if any(key.startswith(prefixes[1:]) for key in state_dict):
            raise RuntimeError("Checkpoint mixes shared and separate WeightMLP keys.")

        legacy = {key[len(legacy_prefix):]: state_dict[key] for key in legacy_keys}
        converted = {}
        for index, branch in enumerate(("weight_mlp_f", "weight_mlp_l")):
            expected = getattr(self, branch).state_dict()
            if set(legacy) != set(expected):
                raise RuntimeError(
                    "Cannot migrate WeightMLP: source and target architectures differ. "
                    "For the old linear checkpoint, set non_linear_weight_mlp=False. "
                    f"Missing keys: {sorted(set(expected) - set(legacy))}; "
                    f"unexpected keys: {sorted(set(legacy) - set(expected))}."
                )
            for suffix, target in expected.items():
                value = legacy[suffix]
                if suffix in ("output_layer.weight", "output_layer.bias"):
                    if value.ndim == 0 or value.shape[0] != 2:
                        raise RuntimeError(
                            f"Expected two output rows in weight_mlp.{suffix}, "
                            f"got {tuple(value.shape)}."
                        )
                    value = value[index:index + 1]
                if value.shape != target.shape:
                    raise RuntimeError(
                        f"Cannot migrate weight_mlp.{suffix} to {branch}.{suffix}: "
                        f"shape {tuple(value.shape)} != {tuple(target.shape)}. "
                        "Check weight_mlp_kwargs, including sin_emb_dim."
                    )
                # Copy Fourier features too, and give the two branches independent storage.
                converted[f"{branch}.{suffix}"] = value.clone()

        # Change the checkpoint only after both branches have passed validation.
        for key in legacy_keys:
            del state_dict[key]
        state_dict.update(converted)

    def forward(self, graph, s, t):
        """Return the coordinate and lattice endpoints under keys f/l."""
        return self.pi(graph, s, t)

    # def weighted_loss(self, losses, s, t):
    #     """Flow Maps log weights, one column each for coordinates and lattice."""
    #     if not self.use_weight_mlp:
    #         return {key: losses[key].mean() for key in ("f", "l")}
    #     weight = self.weight_mlp(s, t)
    #     return {
    #         key: (torch.exp(-weight[:, i]) * losses[key] + weight[:, i]).mean()
    #         for i, key in enumerate(("f", "l"))
    #     }
    def weighted_loss(self, losses, s, t):
        """Flow Maps log weights, one column each for coordinates and lattice."""
        if not self.use_weight_mlp:
            return {key: losses[key].mean() for key in ("f", "l")}
        weight_f = self.weight_mlp_f(s, t).squeeze(-1)
        weight_l = self.weight_mlp_l(s, t).squeeze(-1)
        weighted_loss = {}
        for i, key in enumerate(("f", "l")):
            if key == "f":
                weight = weight_f
            else:
                weight = weight_l
            weighted_loss[key] = (torch.exp(-weight) * losses[key] + weight).mean()
        return weighted_loss

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
        if self.use_weight_mlp:
            loss = (
                self.lambda_f * (weighted_tangent["f"] + weighted_lagrange["f"])
                + self.lambda_l * (weighted_tangent["l"] + weighted_lagrange["l"])
            )
        else:
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
                f"{stage}/lagrange_contribution_f": self.lambda_f * (1.0 if self.use_weight_mlp else self.beta_f) * weighted_lagrange["f"],
                f"{stage}/lagrange_contribution_l": self.lambda_l * (1.0 if self.use_weight_mlp else self.beta_l) * weighted_lagrange["l"],
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

    def on_train_start(self):
        # Extend the restored cosine horizon without changing the current LR,
        # Adam moments, weight decay, or scheduler progress.
        scheduler = self.lr_schedulers()
        scheduler._schedulers[1].T_max = (
            int(self.trainer.estimated_stepping_batches) - scheduler._milestones[0]
        )

    def on_train_batch_start(self, batch, batch_idx):
        group = self.trainer.optimizers[0].param_groups[0]
        self.log_dict(
            {"train/lr": group["lr"], "train/weight_decay": group["weight_decay"]},
            on_step=True, on_epoch=False,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (
                self.ema_pi is not None
                and self.trainer.current_epoch > self.hparams.ema_start
        ):
            self.ema_pi.update_parameters(self.pi)

    @torch.inference_mode()
    def sample(self, batch, n_steps=1, *, pc=True):
        """Sample from 0 to 1, with optional Langevin correction."""
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
        # return self.flow_map.sample_euler(graph0, pi, n_steps=n_steps, t_max_clip=0.99, pc=True, pc_steps=2)
        pc = False
        return self.flow_map.sample(graph0, pi, n_steps=n_steps, t_max_clip=1, pc=pc, pc_steps=3)

    def configure_optimizers(self):
        # parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        # opt = torch.optim.AdamW(
        #     parameters,
        #     lr=self.lr, amsgrad=True, foreach=True, weight_decay=1e-2,
        # )
        parameters = [
            *self.pi.parameters(),
            *(self.weight_mlp_f.parameters() if self.use_weight_mlp else []),
            *(self.weight_mlp_l.parameters() if self.use_weight_mlp else []),
                ]
        opt = torch.optim.AdamW(
                parameters,
                lr=self.lr,
                amsgrad=True,
                foreach=True,
                weight_decay=1e-2,
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
