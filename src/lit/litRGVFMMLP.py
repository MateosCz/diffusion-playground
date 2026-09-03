"""Lightning wrapper for training an RG-VFM endpoint-prediction MLP."""

from typing import Any

import lightning as L
import torch
from torch import nn

from src.flow_matching.rg_vfm import RGVFM


class LitRGVFMMLP(L.LightningModule):
    """Train a time-conditioned MLP with the RG-VFM objective.

    Each batch contains terminal data samples ``x_T``. RG-VFM samples a prior
    state and an intermediate state ``x_t``; the network then regresses
    ``x_T`` from ``(t, x_t)`` using squared geodesic distance.
    """

    def __init__(
        self,
        model: nn.Module,
        rg_vfm: RGVFM,
        flow_kwargs: dict[str, Any],
        batch_size: int,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")

        self.model = model
        if self.model.with_residual_position:
            nn.init.normal_(
                self.model.output_layer.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(self.model.output_layer.bias)
        self.rg_vfm = rg_vfm
        self.flow_kwargs = dict(flow_kwargs)
        self.batch_size = batch_size
        self.lr = lr
        self._train_loss: list[torch.Tensor] = []
        self.save_hyperparameters(ignore=["model", "rg_vfm"])

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Predict the terminal state ``x_T`` using RG-VFM's time-first API."""
        return self.model(t, x_t)

    def _prepare_batch(self, batch: Any) -> torch.Tensor:
        """Extract a floating-point data tensor from a standard loader batch."""
        if isinstance(batch, (tuple, list)):
            if len(batch) != 1:
                raise ValueError(
                    "RG-VFM expects an unlabeled tensor batch or a one-item batch"
                )
            batch = batch[0]
        if not isinstance(batch, torch.Tensor):
            raise TypeError(
                "RG-VFM MLP batches must be tensors, "
                f"got {type(batch).__name__}"
            )

        parameter = next(self.model.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        return batch.to(dtype=dtype)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x_data = self._prepare_batch(batch)
        t, x_t, target_x_T = self.rg_vfm.sample_training_pair(
            x_data,
            time_distribution=self.flow_kwargs.get(
                "time_distribution",
                "uniform",
            ),
            t_min=self.flow_kwargs.get("t_min", 0.0),
            constant_time=self.flow_kwargs.get("constant_time", 0.5),
        )
        pred_x_T = self(t, x_t)
        loss = self.rg_vfm.loss(
            pred_x_T,
            target_x_T,
            x_t=x_t,
            t=t,
        )

        # A model that returns x_t induces a zero vector field. On periodic
        # data this identity-flow baseline can sit deceptively close to the
        # Bayes endpoint loss, so log the gain over it explicitly.
        identity_loss = self.rg_vfm.loss(
            x_t,
            target_x_T,
            x_t=x_t,
            t=t,
        )
        if self.rg_vfm.support == "extrinsic":
            endpoint_movement = self.rg_vfm.manifold.ambient_distance(
                pred_x_T,
                x_t,
            ).mean()
        else:
            endpoint_movement = self.rg_vfm.manifold.distance(
                pred_x_T,
                x_t,
            ).mean()

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/identity_loss": identity_loss,
                f"{stage}/loss_gain_over_identity": identity_loss - loss.detach(),
                f"{stage}/endpoint_movement": endpoint_movement,
            },
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage in ("train", "val"),
            batch_size=x_data.shape[0],
        )
        if stage == "val":
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=x_data.shape[0],
            )
        return loss

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

    @torch.inference_mode()
    def sample(
        self,
        n_samples: int,
        *,
        n_steps: int = 100,
        return_trajectory: bool = False,
    ):
        """Draw samples from the trained RG-VFM model."""
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        parameter = next(self.model.parameters())
        x_0 = self.rg_vfm.sample_prior(
            (n_samples, self.rg_vfm.model_dim),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.rg_vfm.sample(
            self.model,
            x_0,
            n_steps=n_steps,
            return_trajectory=return_trajectory,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


__all__ = ["LitRGVFMMLP"]
