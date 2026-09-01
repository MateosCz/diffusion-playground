"""Distribution-level validation for generated 2D checkerboard samples."""

import lightning as L
import torch


def checkerboard_distribution_metrics(
    points: torch.Tensor,
    *,
    num_rows: int = 4,
    bins: int = 16,
) -> dict[str, torch.Tensor]:
    """Return valid-tile rate and histogram TV for points in ``[0, 1)^2``."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"points must have shape (batch, 2), got {points.shape}")
    if num_rows < 1:
        raise ValueError("num_rows must be positive")
    if bins < 1 or bins % num_rows != 0:
        raise ValueError("bins must be positive and divisible by num_rows")

    wrapped = torch.remainder(points, 1.0)
    tile = torch.floor(wrapped * num_rows).long().clamp(0, num_rows - 1)
    valid_tile_rate = (
        (tile[:, 0] + tile[:, 1]).remainder(2) == 0
    ).to(points.dtype).mean()

    bin_index = torch.floor(wrapped * bins).long().clamp(0, bins - 1)
    flat_index = bin_index[:, 0] * bins + bin_index[:, 1]
    observed = torch.bincount(flat_index, minlength=bins * bins).to(points.dtype)
    observed = observed / observed.sum().clamp_min(1)

    one_dimensional_bins = torch.arange(bins, device=points.device)
    tile_index = one_dimensional_bins // (bins // num_rows)
    valid_bins = (
        tile_index[:, None] + tile_index[None, :]
    ).remainder(2) == 0
    target = valid_bins.flatten().to(points.dtype)
    target = target / target.sum()
    histogram_tv = 0.5 * torch.abs(observed - target).sum()

    return {
        "valid_tile_rate": valid_tile_rate,
        "histogram_tv": histogram_tv,
    }


class CheckerboardGenerationMetrics(L.Callback):
    """Periodically sample a model and log checkerboard distribution metrics."""

    def __init__(
        self,
        *,
        num_rows: int = 4,
        bins: int = 16,
        n_samples: int = 4_096,
        n_steps: int = 100,
        every_n_epochs: int = 25,
    ) -> None:
        super().__init__()
        if n_samples < 1 or n_steps < 1 or every_n_epochs < 1:
            raise ValueError("sample, step and epoch counts must be positive")
        self.num_rows = num_rows
        self.bins = bins
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not hasattr(pl_module, "sample"):
            raise TypeError("checkerboard generation metrics require sample()")

        generated = pl_module.sample(
            self.n_samples,
            n_steps=self.n_steps,
            return_trajectory=False,
        )
        if not isinstance(generated, torch.Tensor):
            raise TypeError("sample() must return a tensor when trajectory is disabled")
        metrics = checkerboard_distribution_metrics(
            generated,
            num_rows=self.num_rows,
            bins=self.bins,
        )
        pl_module.log(
            "val_generated_valid_tile_rate",
            metrics["valid_tile_rate"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            "val_generated_tv",
            metrics["histogram_tv"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


__all__ = [
    "CheckerboardGenerationMetrics",
    "checkerboard_distribution_metrics",
]
