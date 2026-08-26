"""Riemannian Flow Matching (RFM).

This module implements simulation-free conditional flow matching along
shortest geodesics.  A flow starts from ``x0`` sampled from a simple base
distribution and reaches a data sample ``x1`` at ``total_time``.

The implementation only relies on PyTorch.  New manifolds can be supported by
implementing :class:`BaseManifold`; Euclidean space and the flat torus used by
the periodic-coordinate experiments in this repository are included below.
"""

from abc import ABC, abstractmethod
import math
from typing import Callable, Literal, Optional, Sequence

import torch

from src.diffusion.base import BaseDiffusion


class BaseManifold(ABC):
    """Minimal manifold interface required by RFM.

    Points and tangent vectors use the same ambient tensor representation.
    The first tensor dimension is treated as the batch dimension.
    """

    @abstractmethod
    def exp_map(self, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Map a tangent vector at ``x`` back to the manifold."""
        raise NotImplementedError

    @abstractmethod
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the shortest tangent displacement from ``x`` to ``y``."""
        raise NotImplementedError

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project ambient points onto the manifold."""
        return x

    def project_tangent(
        self,
        x: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        """Project an ambient vector onto the tangent space at ``x``."""
        return tangent

    def inner(
        self,
        x: torch.Tensor,
        tangent_a: torch.Tensor,
        tangent_b: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian inner product, reduced over the ambient dimension."""
        return torch.sum(tangent_a * tangent_b, dim=-1)

    @abstractmethod
    def sample_prior(
        self,
        shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample the base distribution on the manifold."""
        raise NotImplementedError


class EuclideanManifold(BaseManifold):
    """Euclidean space with a standard-normal base distribution."""

    def exp_map(self, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return x + tangent

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def sample_prior(
        self,
        shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(tuple(shape), device=device, dtype=dtype)


class FlatTorusManifold(BaseManifold):
    """Flat torus represented in ``[center-period/2, center+period/2)``."""

    def __init__(self, period: float = 2 * math.pi, center: float = 0.0):
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self.period = period
        self.center = center

    def project(self, x: torch.Tensor) -> torch.Tensor:
        lower = self.center - self.period / 2
        return torch.remainder(x - lower, self.period) + lower

    def exp_map(self, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return self.project(x + tangent)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        half_period = self.period / 2
        return torch.remainder(y - x + half_period, self.period) - half_period

    def sample_prior(
        self,
        shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        lower = self.center - self.period / 2
        samples = torch.rand(tuple(shape), device=device, dtype=dtype)
        return samples * self.period + lower


class RiemannianFlowMatching(BaseDiffusion):
    """Conditional flow matching along shortest Riemannian geodesics.

    For paired base/data samples ``(x0, x1)``, the conditional path is

    ``x_t = Exp_x0((t / total_time) Log_x0(x1))``.

    The regression target is the velocity of this path at ``x_t``.  It can be
    evaluated without differentiating through the path as

    ``u_t = Log_xt(x1) / (total_time - t)``.

    Parameters
    ----------
    manifold:
        Geometry and base distribution.  Defaults to Euclidean space.
    total_time:
        End time of the generative flow.
    time_eps:
        Training times stay this far away from the singular endpoint.
    normalize_loss:
        Divide the squared Riemannian norm by the ambient dimension.  This
        makes the Euclidean objective identical to ``torch`` MSE.
    """

    def __init__(
        self,
        manifold: Optional[BaseManifold] = None,
        total_time: float = 1.0,
        time_eps: float = 1e-5,
        normalize_loss: bool = True,
    ):
        super().__init__()
        if total_time <= 0:
            raise ValueError(f"total_time must be positive, got {total_time}")
        if not 0 <= time_eps < total_time:
            raise ValueError(
                f"time_eps must be in [0, total_time), got {time_eps}"
            )
        self.manifold = manifold or EuclideanManifold()
        self.total_time = total_time
        self.time_eps = time_eps
        self.normalize_loss = normalize_loss

    def sample_forward(
        self,
        x1: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        t_dist_kw: Literal["uniform", "quadratic", "constant"] = "uniform",
        constant_t: float = 0.5,
        return_time: bool = False,
        t_min: float = 0.0,
        ts: Optional[torch.Tensor] = None,
    ):
        """Sample a conditional geodesic state and its target velocity.

        ``x1`` contains data samples.  When ``x0`` is omitted it is sampled
        from the manifold's base distribution.  A supplied ``ts`` must contain
        one time per batch item, matching the shared-time convention used by
        the other diffusion classes in this repository.
        """
        if x1.ndim < 2:
            raise ValueError(
                f"x1 must include batch and feature dimensions, got {x1.shape}"
            )
        if x0 is None:
            x0 = self.sample_prior(x1)
        elif x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}"
            )
        else:
            x0 = x0.to(device=x1.device, dtype=x1.dtype)

        t = self._sample_time(
            x1,
            t_dist_kw=t_dist_kw,
            constant_t=constant_t,
            t_min=t_min,
            ts=ts,
        )
        expanded_t = self._expand_time_to_x(t, x1)
        initial_tangent = self.manifold.log_map(x0, x1)
        x_t = self.manifold.exp_map(
            x0,
            expanded_t / self.total_time * initial_tangent,
        )
        target = self.conditional_vector_field(x_t, x1, expanded_t)

        if return_time:
            return x_t, target, t
        return x_t, target

    def conditional_vector_field(
        self,
        x_t: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the closed-form conditional RFM target at ``x_t``."""
        if x_t.shape != x1.shape:
            raise ValueError(
                f"x_t and x1 must have the same shape, got {x_t.shape} and {x1.shape}"
            )
        t = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
        if t.ndim == 0:
            t = t.expand(x_t.shape[0]).unsqueeze(-1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim == 2:
            if t.shape != (x_t.shape[0], 1):
                raise ValueError(
                    f"time must have shape ({x_t.shape[0]}, 1), got {tuple(t.shape)}"
                )
            t = self._expand_time_to_x(t, x_t)
        remaining_time = self.total_time - t
        if torch.any(remaining_time <= 0):
            raise ValueError("RFM targets are undefined at or after total_time")
        target = self.manifold.log_map(x_t, x1) / remaining_time
        return self.manifold.project_tangent(x_t, target)

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean squared Riemannian error between predicted and target flow.

        ``x_t`` is optional for the included Euclidean and flat-torus
        geometries because their metrics are constant.  It must be supplied
        for a custom manifold whose metric or tangent projection depends on
        position.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
            )
        if x_t is None:
            if not isinstance(
                self.manifold,
                (EuclideanManifold, FlatTorusManifold),
            ):
                raise ValueError("x_t is required for a custom manifold loss")
            x_t = target
        elif x_t.shape != target.shape:
            raise ValueError(
                f"x_t and target must have the same shape, got {x_t.shape} and {target.shape}"
            )
        error = self.manifold.project_tangent(x_t, pred - target)
        loss = self.manifold.inner(x_t, error, error).mean()
        if self.normalize_loss:
            loss = loss / target.shape[-1]
        return loss

    def sample_prior(
        self,
        shape_or_like: Sequence[int] | torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Sample the base distribution with an explicit shape or like a tensor."""
        if isinstance(shape_or_like, torch.Tensor):
            shape = shape_or_like.shape
            device = shape_or_like.device
            dtype = shape_or_like.dtype
        else:
            shape = shape_or_like
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
        return self.manifold.sample_prior(shape, device=device, dtype=dtype)

    def forward_step(
        self,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        """Advance one manifold-aware Euler step in generative time."""
        tangent = self.manifold.project_tangent(x_t, pred)
        return self.manifold.exp_map(x_t, tangent * dt)

    @torch.inference_mode()
    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        """Move one Euler step from data time back toward base time."""
        return self.forward_step(x_t, pred, -dt)

    @torch.inference_mode()
    def sample_backward(
        self,
        x0: torch.Tensor,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_steps: int = 100,
        sample_trajectory: bool = False,
    ):
        """Generate data by integrating the learned flow from base to data.

        ``vector_field`` follows this repository's data-first convention and is
        called as ``vector_field(x_t, t)``.  Despite the historical
        ``sample_backward`` name used by diffusion classes, RFM is integrated
        from time zero to ``total_time``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        if x0.ndim < 2:
            raise ValueError(
                f"x0 must include batch and feature dimensions, got {x0.shape}"
            )

        x_t = self.manifold.project(x0)
        trajectory = [x_t.clone()] if sample_trajectory else None
        dt = self.total_time / n_steps

        for step in range(n_steps):
            t = torch.full(
                (x_t.shape[0], 1),
                step * dt,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            pred = vector_field(x_t, t)
            if pred.shape != x_t.shape:
                raise ValueError(
                    f"vector_field must return shape {x_t.shape}, got {pred.shape}"
                )
            x_t = self.forward_step(x_t, pred, dt)
            if trajectory is not None:
                trajectory.append(x_t.clone())

        if trajectory is not None:
            return torch.stack(trajectory, dim=0)
        return x_t

    def _sample_time(
        self,
        x: torch.Tensor,
        t_dist_kw: Literal["uniform", "quadratic", "constant"],
        constant_t: float,
        t_min: float,
        ts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not 0 <= t_min < self.total_time - self.time_eps:
            raise ValueError(
                "t_min must be non-negative and smaller than total_time - time_eps"
            )

        if ts is not None:
            t = torch.as_tensor(ts, device=x.device, dtype=x.dtype)
            if t.ndim == 0:
                t = t.expand(x.shape[0]).unsqueeze(-1)
            elif t.ndim == 1:
                t = t.unsqueeze(-1)
        elif t_dist_kw == "constant":
            t = torch.full(
                (x.shape[0], 1),
                constant_t,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            u = torch.rand((x.shape[0], 1), device=x.device, dtype=x.dtype)
            if t_dist_kw == "quadratic":
                u = u.square()
            elif t_dist_kw != "uniform":
                raise ValueError(f"Unknown t_dist_kw: {t_dist_kw}")
            t_max = self.total_time - self.time_eps
            t = u * (t_max - t_min) + t_min

        if t.shape != (x.shape[0], 1):
            raise ValueError(
                f"time must have shape ({x.shape[0]}, 1), got {tuple(t.shape)}"
            )
        if torch.any(t < 0) or torch.any(t >= self.total_time):
            raise ValueError(f"time values must be in [0, {self.total_time})")
        return t

    @staticmethod
    def _expand_time_to_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], *([1] * (x.ndim - 1)))


RFM = RiemannianFlowMatching


__all__ = [
    "BaseManifold",
    "EuclideanManifold",
    "FlatTorusManifold",
    "RiemannianFlowMatching",
    "RFM",
]
