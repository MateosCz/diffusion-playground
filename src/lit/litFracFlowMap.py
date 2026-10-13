from typing import Any

import lightning as L
import torch
from torch import nn

from src.flow_matching.rg_vfm import RGVFM
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.lit.callbacks import last_checkpoint

class LitFracFlowMapNN(L.LightningModule):
    def __init__(
        self,
        pi:nn.Module, # endpoint partial denoiser
        eta: float, # diagonal fraction (0,1]
        rg_vfm: RGVFM,
        flow_map: VFMFlowMap,
        flow_kwargs: dict[str, Any],
        rg_vfm_kwargs: dict[str, Any], # for logging
        nn_kwargs: dict[str, Any], # for logging
        experiment_name_timestamp: str, # for logging
        batch_size: int,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.pi = pi
        self.eta = eta
        if self.pi.with_residual_position:
            nn.init.normal_(
                self.pi.output_layer.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(self.pi.output_layer.bias)
        self.rg_vfm = rg_vfm
        self.flow_kwargs = flow_kwargs
        self.rg_vfm_kwargs = rg_vfm_kwargs
        self.nn_kwargs = nn_kwargs
        self.flow_map = flow_map
        self.batch_size = batch_size
        self.lr = lr
        self._train_loss: list[torch.Tensor] = []
        self.save_hyperparameters(ignore=["pi", "rg_vfm", "flow_map"])



    def forward(self, x_s, s, t) -> torch.Tensor:
        """the output of the endpoint partial denoiser pi_s,t(x_s)"""
        return self.pi.forward(x_s, s, t)

    def _prepare_batch(self, batch: Any) -> torch.Tensor:
        """Extract a floating-point data tensor from a standard loader batch."""
        if isinstance(batch, (tuple, list)):
            if len(batch) != 1:
                raise ValueError(
                    "RG-VFM expects an unlabeled tensor batch or a one-item batch"
                )
            batch = batch[0]

        parameter = next(self.pi.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        return batch.to(dtype=dtype)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        """shared step for training, validation, and test
        follow the algorithm 2 in CFM paper.
        
        """
        x_data = self._prepare_batch(batch) # make the datatype of the batch the same as the model
        # Keep at least one sample in each branch.
        M = x_data.shape[0]
        if M <= 1:
            raise ValueError("batch must contain at least 2 samples")
        M_d = min(max(int(self.eta * M), 1), M - 1)
        M_o = M - M_d
        x_d, x_o = x_data.split([M_d, M_o], dim=0) #x_d: diagonal samples, x_o: off-diagonal samples
        
        # Tangent samples
        ti, x_tj, target_x_1j = self.rg_vfm.sample_training_pair(
            x_d,
            time_distribution=self.flow_kwargs.get("time_distribution", "uniform"),
            t_min=self.flow_kwargs.get("t_min", 0.0),
        )
        loss = self.flow_map.tangent_loss(target_x_1j, x_tj, ti, self.pi)

        # Compare the tangent prediction with the identity-flow baseline.
        # keep the detailed validation of the tangent loss from flow matching model
        pred_x_1j = self.pi(x_tj, ti, ti).detach()
        identity_loss = self.rg_vfm.loss(x_tj, target_x_1j, t=ti)
        metrics = {
            f"{stage}/identity_loss": identity_loss,
            f"{stage}/loss_gain_over_identity": identity_loss - loss.detach(),
        }
        if self.rg_vfm.support == "extrinsic":
            metrics[f"{stage}/endpoint_movement"] = self.rg_vfm.manifold.ambient_distance(
                pred_x_1j, x_tj,
            ).mean()

        


        # Lagrange samples
        sj, x_sj, target_x_1j = self.rg_vfm.sample_training_pair(
            x_o,
            time_distribution=self.flow_kwargs.get("time_distribution", "uniform"),
            t_min=self.flow_kwargs.get("t_min", 0.0),
        )
        # Sample t_j uniformly between s_j and total_time.
        tj = sj + torch.rand_like(sj) * (self.rg_vfm.total_time - sj)
        lagrange_loss = self.flow_map.lagrange_loss(x_sj, sj, tj, self.pi)
        metrics[f"{stage}/lagrange_loss"] = lagrange_loss
        metrics[f"{stage}/tangent_loss"] = loss
        metrics[f"{stage}/total_loss"] = loss + lagrange_loss
        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage in ("train", "val"),
            batch_size=x_data.shape[0],
        )
        if stage == "val":
            self.log(
                "val_tangent_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=x_data.shape[0],
            )
            self.log(
                "val_lagrange_loss",
                lagrange_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=x_data.shape[0],
            )
            self.log(
                "val_total_loss",
                loss + lagrange_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=x_data.shape[0],
            )
        
        return loss + lagrange_loss

    def training_step(self, batch: Any, batch_idx: int = 0) -> torch.Tensor:
        del batch_idx
        loss = self._shared_step(batch, "train")
        self._train_loss.append(loss.detach())
        return loss

    def validation_step(self, batch: Any, batch_idx: int = 0) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int = 0) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self) -> None:
        if not self._train_loss:
            return
        losses = torch.stack(self._train_loss)
        self.log_dict(
            {
                "train/loss_mean": losses.mean(),
                "train/loss_var": losses.var(unbiased=False),
                "train/loss_std": losses.std(unbiased=False),
            },
            on_step=False,
            on_epoch=True,
        )
        self._train_loss.clear()

    # @torch.inference_mode()
    # def sample(
    #     self,
    #     n_samples: int,
    #     *,
    #     n_steps: int = 100,
    #     return_trajectory: bool = False,
    # ):
    #     """Draw samples from the trained RG-VFM model."""
    #     if n_samples < 1:
    #         raise ValueError(f"n_samples must be positive, got {n_samples}")
    #     parameter = next(self.model.parameters())
    #     x_0 = self.rg_vfm.sample_prior(
    #         (n_samples, self.rg_vfm.model_dim),
    #         device=parameter.device,
    #         dtype=parameter.dtype,
    #     )
    #     return self.rg_vfm.sample(
    #         self.model,
    #         x_0,
    #         n_steps=n_steps,
    #         return_trajectory=return_trajectory,
    #     )

    @torch.inference_mode()
    def sample(self, num_samples: int, *,n_steps: int = 100,return_trajectory: bool = False):
        """Sample from the trained flow-map model by applying n_steps flow-map updates from 0 to total_time.

        Return continuous model-space states of shape (num_samples, model_dim).
        """
        reference = next(self.pi.parameters())
        time_steps = torch.linspace(
            0, self.rg_vfm.total_time, n_steps + 1,
            device=reference.device, dtype=reference.dtype,
        )
        x = self.rg_vfm.sample_prior(
            (num_samples, self.rg_vfm.model_dim),
            device=reference.device,
            dtype=reference.dtype,
        )
        for s, t in zip(time_steps[:-1], time_steps[1:]):
            x = self.flow_map.Xst(x, s, t, self.pi)
        return x




    def configure_optimizers(self):
        return torch.optim.Adam(self.pi.parameters(), lr=self.lr)
