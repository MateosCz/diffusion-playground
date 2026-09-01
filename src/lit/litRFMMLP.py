"""Lightning wrapper for Riemannian velocity matching with an MLP."""

from typing import Any

import lightning as L
import torch
from torch import nn

from src.flow_matching.rfm import RFM


class LitRFMMLP(L.LightningModule):
    """Train a time-conditioned MLP against conditional geodesic velocity.

    In addition to the actual loss, the module logs a zero-velocity baseline.
    This exposes the analogue of the identity-flow degeneration: a useful
    model must beat the baseline and predict a non-trivial vector field.
    """

    def __init__(
        self,
        model: nn.Module,
        rfm: RFM,
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
        self.rfm = rfm
        self.flow_kwargs = dict(flow_kwargs)
        self.batch_size = batch_size
        self.lr = lr
        self._train_loss: list[torch.Tensor] = []
        self.save_hyperparameters(ignore=["model", "rfm"])

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.model(t, x_t)

    def _prepare_batch(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            if len(batch) != 1:
                raise ValueError(
                    "RFM expects an unlabeled tensor batch or a one-item batch"
                )
            batch = batch[0]
        if not isinstance(batch, torch.Tensor):
            raise TypeError(
                f"RFM MLP batches must be tensors, got {type(batch).__name__}"
            )

        parameter = next(self.model.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        return batch.to(dtype=dtype)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x_data = self._prepare_batch(batch)
        t, x_t, target_velocity = self.rfm.sample_training_pair(
            x_data,
            time_distribution=self.flow_kwargs.get("time_distribution", "uniform"),
            t_min=self.flow_kwargs.get("t_min", 0.0),
            constant_time=self.flow_kwargs.get("constant_time", 0.5),
        )
        pred_velocity = self(t, x_t)
        loss = self.rfm.loss(
            pred_velocity,
            target_velocity,
            x_t=x_t,
            t=t,
        )

        zero_velocity_loss = self.rfm.loss(
            torch.zeros_like(target_velocity),
            target_velocity,
            x_t=x_t,
            t=t,
        )
        pred_velocity_rms = pred_velocity.square().mean().sqrt()
        target_velocity_rms = target_velocity.square().mean().sqrt()
        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/zero_velocity_loss": zero_velocity_loss,
            f"{stage}/loss_gain_over_zero": zero_velocity_loss - loss.detach(),
            f"{stage}/pred_velocity_rms": pred_velocity_rms,
            f"{stage}/target_velocity_rms": target_velocity_rms,
        }
        self.log_dict(
            metrics,
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
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        parameter = next(self.model.parameters())
        x_0 = self.rfm.sample_prior(
            (n_samples, self.rfm.manifold.intrinsic_dim),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.rfm.sample(
            self.model,
            x_0,
            n_steps=n_steps,
            return_trajectory=return_trajectory,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


__all__ = ["LitRFMMLP"]
