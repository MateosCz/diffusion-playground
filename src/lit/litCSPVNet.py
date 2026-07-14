import torch
from torch import nn
import lightning as L
from torch_geometric.data import Data
from typing import Optional, Tuple
from src.metrics.csp import CSPMetrics
from src.dataLib.data_util import kldm_output_to_structures_batch
from src.diffusion.kldm import KLDM
from src.dataLib.data_util import PyGData_to_Structure
import torch_geometric.transforms as PyGT

def time_loss_weight(t: torch.Tensor, total_time: float) -> torch.Tensor:
    t_norm = (t / total_time).clamp(0.0, 1.0)
    return (1.0 - t_norm).clamp_min(0.05)


def weighted_score_loss(
    pred_score: torch.Tensor,
    target_score: torch.Tensor,
    t: torch.Tensor,
    total_time: float,
) -> torch.Tensor:
    """Time-weighted MSE between predicted and target scores.

    ``t`` carries one time per row of ``pred_score`` (per-node or per-graph);
    the weight down-weights the noisiest (large-t) samples.
    """
    weight = time_loss_weight(t, total_time)
    while weight.ndim < pred_score.ndim:
        weight = weight.unsqueeze(-1)
    return torch.mean(weight * (pred_score - target_score) ** 2)


class LitCSPVNet(L.LightningModule):
    """
    LightningModule that lifts ``CSPVNet`` (a joint lattice + fractional-coordinate
    score network) into a trainable model on top of the ``KLDM`` diffusion.

    Each batch is a PyG ``Data``/``Batch`` carrying (at least):
      - ``pos`` / ``x`` : (N_total, 3)   fractional coordinates
      - ``l``           : (num_graph, 6) lattice parameters ([log-lengths, tan-angles])
      - ``h``           : atom types / features consumed by ``CSPVNet``
      - ``batch``       : (N_total,)     graph membership
      - ``edge_index``  : (2, E_total)   connectivity

    ``KLDM.sample_forward`` produces the noised state and the training targets:
    a single per-graph diffusion time is shared between the lattice and the
    fractional coordinates, matching the single-time conditioning of ``CSPVNet``.
    """

    def __init__(
        self,
        model: nn.Module,
        kldm: KLDM,
        diffusion_kwargs: dict,
        sample_kwargs: dict,
        batch_size: int,
        lr: float = 1e-3,
        lambda_l: float = 1.0,
        lambda_f: float = 1.0,
        transform: PyGT.Compose = None,
        matcher_kwargs: Optional[dict] = None,
        n_t_bins: int = 10,
        t_bucket_repeats: int = 4,
    ):
        super().__init__()
        self.model = model
        self.kldm = kldm
        self.diffusion_kwargs = diffusion_kwargs
        self.sample_kwargs = sample_kwargs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_l = lambda_l
        self.lambda_f = lambda_f
        self.transform = transform
        # per-t-bucket validation diagnostics
        self.n_t_bins = n_t_bins
        self.t_bucket_repeats = t_bucket_repeats
        self._train_loss = []
        self._val_metrics = CSPMetrics(**(matcher_kwargs or {}))
        self._test_metrics = CSPMetrics(**(matcher_kwargs or {}))
        self._reset_t_buckets()
        self.save_hyperparameters(ignore=["model", "kldm"])

    def forward_from_data(
        self, graph: Data, vt: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run ``CSPVNet`` on a PyG batch and return ``(score_l, score_f)``.

        The tuple layout matches the ``score_fn`` signature expected by
        ``KLDM.sample_backward``, so the same module can be reused for sampling.

        Parameters
        ----------
        graph : PyG Data/Batch with ``pos``/``x``, ``l``, ``h``, ``batch``, ``edge_index``
        vt    : (N_total, dim)  node velocities
        t     : (num_graph, 1)  per-graph diffusion time

        Returns
        -------
        score_l : (num_graph, 6) lattice score
        score_f : (N_total, dim) fractional-coordinate score
        """
        out = self.model(
            t=t,
            pos=graph.pos,
            vt=vt,
            h=graph.h,
            l=graph.l,
            batch=graph.batch,
            edge_index=graph.edge_index,
        )
        return out["l"], out["v"]

    def _shared_step(self, batch: Data, stage: str) -> torch.Tensor:
        (noised_v, noised_graph), score_dict, t_dict = self.kldm.sample_forward(
            graph0=batch,
            t_min=self.diffusion_kwargs.get("t_min", 5e-3),
            t_dist_kw=self.diffusion_kwargs.get("t_dist_kw", "uniform"),
            shared_time=self.diffusion_kwargs.get("shared_time", True),
            return_time=True,
        )

        # single shared time -> condition the network on it
        t_graph = t_dict["f"]                       # (num_graph, 1)
        pred_l, pred_f = self.forward_from_data(noised_graph, noised_v, t_graph)

        target_l = score_dict["l"].squeeze(-1)      # (num_graph, 6)
        target_f = score_dict["f"]                  # (N_total, dim)

        total_time = self.kldm.total_time
        t_graph_l = t_dict["l"]                      # (num_graph, 1)
        t_node_f = t_dict["f"][batch.batch]          # (N_total, 1)

        loss_l = weighted_score_loss(pred_l, target_l, t_graph_l, total_time)
        loss_f = weighted_score_loss(pred_f, target_f, t_node_f, total_time)
        loss = self.lambda_l * loss_l + self.lambda_f * loss_f

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/loss_l": loss_l,
                f"{stage}/loss_f": loss_f,
            },
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    # ------------------------------------------------------------------
    # Per-t-bucket validation error
    # ------------------------------------------------------------------
    # Records the *raw*, un-weighted prediction error split into noise-level
    # bands, so we can see which part of the diffusion trajectory the network
    # fits badly -- decoupled from the training-time ``time_loss_weight``.
    # Lattice: x0-error (parameterization="x0"); coords: score-error.
    def _reset_t_buckets(self) -> None:
        n = self.n_t_bins
        self._t_bucket_l_sqerr = torch.zeros(n)
        self._t_bucket_l_count = torch.zeros(n)
        self._t_bucket_f_sqerr = torch.zeros(n)
        self._t_bucket_f_count = torch.zeros(n)

    def _t_bucket_index(self, t: torch.Tensor) -> torch.Tensor:
        """Map per-sample times (any shape) to bin indices in [0, n_t_bins)."""
        t_min = self.diffusion_kwargs.get("t_min", 5e-3)
        total_time = self.kldm.total_time
        frac = ((t.reshape(-1) - t_min) / (total_time - t_min)).clamp(0.0, 1.0 - 1e-6)
        return (frac * self.n_t_bins).long()

    def _accumulate_t_buckets(
        self,
        t: torch.Tensor,
        sqerr: torch.Tensor,
        sq_accum: torch.Tensor,
        count_accum: torch.Tensor,
    ) -> None:
        idx = self._t_bucket_index(t).cpu()
        sqerr = sqerr.detach().reshape(-1).cpu().float()
        sq_accum.index_add_(0, idx, sqerr)
        count_accum.index_add_(0, idx, torch.ones_like(sqerr))

    @torch.no_grad()
    def _update_t_buckets_from_batch(self, batch: Data) -> None:
        """Freshly noise ``batch`` a few times and bucket the raw error by t."""
        for _ in range(self.t_bucket_repeats):
            (noised_v, noised_graph), score_dict, t_dict = self.kldm.sample_forward(
                graph0=batch,
                t_min=self.diffusion_kwargs.get("t_min", 5e-3),
                t_dist_kw=self.diffusion_kwargs.get("t_dist_kw", "uniform"),
                shared_time=self.diffusion_kwargs.get("shared_time", True),
                return_time=True,
            )
            pred_l, pred_f = self.forward_from_data(noised_graph, noised_v, t_dict["f"])

            target_l = score_dict["l"].squeeze(-1)           # (num_graph, 6)
            target_f = score_dict["f"]                        # (N_total, dim)
            err_l = (pred_l - target_l).pow(2).mean(dim=-1)   # (num_graph,)
            err_f = (pred_f - target_f).pow(2).mean(dim=-1)   # (N_total,)

            t_l = t_dict["l"]                                 # (num_graph, 1)
            t_f = t_dict["f"][batch.batch]                    # (N_total, 1)
            self._accumulate_t_buckets(t_l, err_l, self._t_bucket_l_sqerr, self._t_bucket_l_count)
            self._accumulate_t_buckets(t_f, err_f, self._t_bucket_f_sqerr, self._t_bucket_f_count)

    def _log_t_buckets(self) -> None:
        t_min = self.diffusion_kwargs.get("t_min", 5e-3)
        total_time = self.kldm.total_time
        edges = torch.linspace(t_min, total_time, self.n_t_bins + 1)
        mean_l = self._t_bucket_l_sqerr / self._t_bucket_l_count.clamp_min(1.0)
        mean_f = self._t_bucket_f_sqerr / self._t_bucket_f_count.clamp_min(1.0)

        logs = {}
        for i in range(self.n_t_bins):
            if self._t_bucket_l_count[i] > 0:
                logs[f"val_terr/l_bin{i:02d}"] = mean_l[i]
            if self._t_bucket_f_count[i] > 0:
                logs[f"val_terr/f_bin{i:02d}"] = mean_f[i]
        if logs:
            self.log_dict(logs)

        # human-readable table (bin -> [t_lo, t_hi) mapping is otherwise opaque)
        if self.trainer is not None and self.trainer.is_global_zero:
            print("\n[val per-t error]  bin  [t_lo,  t_hi)   err_l (n)          err_f (n)")
            for i in range(self.n_t_bins):
                lo, hi = edges[i].item(), edges[i + 1].item()
                cl, cf = int(self._t_bucket_l_count[i]), int(self._t_bucket_f_count[i])
                print(
                    f"  {i:02d}: [{lo:5.3f},{hi:5.3f})  "
                    f"l={mean_l[i]:.4e} (n={cl:5d})  f={mean_f[i]:.4e} (n={cf:5d})"
                )

    def training_step(self, batch: Data) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        self._train_loss.append(loss.detach())
        return loss

    def on_validation_epoch_start(self):
        self._val_metrics.reset()
        self._reset_t_buckets()

    def validation_step(self, batch: Data) -> None:
        # raw per-t-bucket error (cheap forward passes on freshly noised data)
        self._update_t_buckets_from_batch(batch)

        transform_lengths, transform_angles, transform_pos = self.transform.transforms[0], self.transform.transforms[1], self.transform.transforms[2]
        target_structures = [PyGData_to_Structure(batch[j], transform_lengths, transform_angles, transform_pos) for j in range(batch.num_graphs)]
        gen_l0, gen_f0 = self.sample(batch)
        self._val_metrics.update(
            kldm_output_to_structures_batch(batch, gen_l0, gen_f0, transform_lengths, transform_angles, transform_pos), target_structures
        )

    def on_validation_epoch_end(self):
        summary = self._val_metrics.summarize()
        self.log_dict({f"val/{k}": v for k, v in summary.items()})
        self._log_t_buckets()

    def on_test_epoch_start(self):
        self._test_metrics.reset()

    def test_step(self, batch: Data) -> None:
        transform_lengths, transform_angles, transform_pos = self.transform.transforms[0], self.transform.transforms[1], self.transform.transforms[2]
        target_structures = [PyGData_to_Structure(batch[j], transform_lengths, transform_angles, transform_pos) for j in range(batch.num_graphs)]
        gen_l0, gen_f0 = self.sample(batch)
        self._test_metrics.update(
            kldm_output_to_structures_batch(batch, gen_l0, gen_f0, transform_lengths, transform_angles, transform_pos), target_structures
        )

    def on_test_epoch_end(self):
        summary = self._test_metrics.summarize()
        self.log_dict({f"test/{k}": v for k, v in summary.items()})
    
    @torch.no_grad()
    def sample(self, batch: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        gen_l0, gen_f0 = self.kldm.sample_backward(
            graphT=batch,
            score_fn=self.forward_from_data,
            **self.sample_kwargs,
        )
        return gen_l0, gen_f0

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def on_before_optimizer_step(self, optimizer) -> None:
        """Neutralize non-finite gradients before the optimizer step.

        Score-matching targets can spike at very small diffusion times; a single
        NaN/Inf gradient would otherwise be written into every weight (and
        gradient-norm clipping cannot recover from an already non-finite grad).
        Replacing non-finite grads with 0 makes that batch a no-op for the
        affected parameters instead of killing the whole run.
        """
        for p in self.model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def on_train_epoch_end(self):
        if not self._train_loss:
            return
        losses = torch.stack(self._train_loss)
        self.log_dict(
            {
                "train/loss_mean": losses.mean(),
                "train/loss_var": losses.var(unbiased=False),
                "train/loss_std": losses.std(unbiased=False),
            }
        )
        self._train_loss.clear()
