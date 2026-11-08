"""Euclidean VFM: Gaussian prior, linear paths and endpoint regression."""

from collections.abc import Sequence

import torch
from torch import nn

from src.flow_matching.base import BaseFlowMatching, TimeDistribution


class VFM(nn.Module):
    """Predict the data endpoint of a linear path in ``R^dim``."""

    # These time helpers do not depend on a manifold.
    batch_time = staticmethod(BaseFlowMatching.batch_time)
    sample_time = BaseFlowMatching.sample_time

    def __init__(
        self,
        dim: int,
        time_eps: float = 1e-5,
        prior_scale: float = 1.0,
        max_loss_weight: float = 100.0,
        is_loss_weighted: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not 0 <= time_eps < 1:
            raise ValueError("time_eps must be in [0, 1)")
        if prior_scale <= 0:
            raise ValueError("prior_scale must be positive")
        self.model_dim = dim
        self.total_time = 1.0
        self.time_eps = float(time_eps)
        self.prior_scale = float(prior_scale)
        self.max_loss_weight = float(max_loss_weight)
        self.is_loss_weighted = is_loss_weighted

    def sample_prior(
        self,
        shape_or_like: Sequence[int] | torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Draw Gaussian states, matching a reference tensor when supplied."""
        if isinstance(shape_or_like, torch.Tensor):
            shape = tuple(shape_or_like.shape)
            device = device if device is not None else shape_or_like.device
            dtype = dtype if dtype is not None else shape_or_like.dtype
        else:
            shape = tuple(shape_or_like)
        if len(shape) != 2 or shape[-1] != self.model_dim:
            raise ValueError(f"prior shape must be (batch, {self.model_dim})")
        return torch.randn(shape, device=device, dtype=dtype) * self.prior_scale

    def sample_training_pair(
        self,
        x_data: torch.Tensor,
        x_0: torch.Tensor | None = None,
        *,
        t: torch.Tensor | None = None,
        time_distribution: TimeDistribution = "uniform",
        t_min: float = 0.0,
        constant_time: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(t, (1-t) * x_0 + t * x_data, x_data)``."""
        self._validate_state(x_data, "x_data")
        if x_0 is None:
            x_0 = self.sample_prior(x_data)
        else:
            BaseFlowMatching._validate_pair(x_0, x_data)
            x_0 = x_0.to(x_data)
        times = self.sample_time(
            x_data, distribution=time_distribution, t_min=t_min,
            constant_time=constant_time, t=t,
        )
        return times, (1 - times) * x_0 + times * x_data, x_data

    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor,
        *, t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sample endpoint MSE, optionally weighted by ``1/(1-t)^2``."""
        self._validate_state(prediction, "prediction")
        BaseFlowMatching._validate_pair(prediction, target)
        times = self.batch_time(0.0 if t is None else t, prediction).squeeze(-1)
        if torch.any(times < 0) or torch.any(times >= 1):
            raise ValueError("loss times must be in [0, 1)")
        errors = (prediction - target).square().mean(dim=-1)
        if self.is_loss_weighted:
            weight = 1.0/(1 - times).square()
            if self.max_loss_weight is not None:
                weight = weight.clamp_max(self.max_loss_weight)
            errors = weight * errors
            
        return errors

    def _validate_state(self, x: torch.Tensor, name: str) -> None:
        BaseFlowMatching._validate_points(x, name)
        if x.shape[-1] != self.model_dim:
            raise ValueError(f"{name} must have {self.model_dim} features")
