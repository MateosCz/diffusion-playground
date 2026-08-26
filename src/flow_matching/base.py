"""Shared contracts for flow-matching methods."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
from torch import nn

from src.manifolds.base import BaseManifold
from src.ode.integrators import BaseODEIntegrator, get_integrator


TimeDistribution = Literal["uniform", "quadratic", "constant"]
FlowModel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseFlowMatching(ABC, nn.Module):
    """Base class for deterministic flow-matching probability paths.

    Flow models and vector fields use the ODE convention ``model(t, x)``.
    Training methods return ``(t, x_t, target)`` so time is explicit and the
    result can be passed directly to a time-conditioned network.
    """

    def __init__(
        self,
        manifold: BaseManifold,
        *,
        total_time: float = 1.0,
        time_eps: float = 1e-5,
        integrator: str | BaseODEIntegrator = "euler",
    ) -> None:
        super().__init__()
        if not isinstance(manifold, BaseManifold):
            raise TypeError("manifold must implement BaseManifold")
        if total_time <= 0:
            raise ValueError(f"total_time must be positive, got {total_time}")
        if not 0 <= time_eps < total_time:
            raise ValueError(
                f"time_eps must be in [0, total_time), got {time_eps}"
            )
        self.manifold = manifold
        self.total_time = float(total_time)
        self.time_eps = float(time_eps)
        self.integrator = get_integrator(integrator)

    @abstractmethod
    def sample_training_pair(
        self,
        x_data: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        *,
        t: Optional[torch.Tensor] = None,
        time_distribution: TimeDistribution = "uniform",
        t_min: float = 0.0,
        constant_time: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(t, x_t, regression_target)`` for training."""
        raise NotImplementedError

    @abstractmethod
    def loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the method-specific flow-matching objective."""
        raise NotImplementedError

    @abstractmethod
    def vector_field(
        self,
        model: FlowModel,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the model-induced tangent vector field."""
        raise NotImplementedError

    def sample_path(
        self,
        x_0: torch.Tensor,
        x_data: torch.Tensor,
        *,
        t: Optional[torch.Tensor | float] = None,
        n_steps: int = 100,
    ):
        """Evaluate a conditional geodesic at ``t`` or on a complete grid.

        A supplied ``t`` returns one state per batch item.  With ``t=None``,
        the return value is ``(times, states)`` and includes both endpoints.
        """
        self._validate_pair(x_0, x_data)
        if t is not None:
            return self._conditional_state(x_0, x_data, t)
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        times = self.time_grid(x_data, n_steps)
        states = torch.stack(
            [self._conditional_state(x_0, x_data, time) for time in times],
            dim=0,
        )
        return times, states

    @torch.inference_mode()
    def sample(
        self,
        model: FlowModel,
        x_0: torch.Tensor,
        *,
        n_steps: int = 100,
        integrator: Optional[str | BaseODEIntegrator] = None,
        return_trajectory: bool = False,
    ):
        """Generate data by solving the learned ODE from base to data time."""
        self._validate_points(x_0, "x_0")
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        solver = self.integrator if integrator is None else get_integrator(integrator)
        times = self.time_grid(x_0, n_steps)

        def field(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.vector_field(model, t, x)

        return solver.integrate(
            field,
            x_0,
            times,
            state_update=self.state_update,
            transport=self.tangent_transport,
            return_trajectory=return_trajectory,
        )

    def state_update(
        self,
        x: torch.Tensor,
        increment: torch.Tensor,
    ) -> torch.Tensor:
        """Apply an ODE increment in the method's state representation."""
        return self.manifold.exp_map(x, increment)

    def tangent_transport(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        """Transport tangents for higher-order ODE integration."""
        return self.manifold.parallel_transport(source, target, tangent)

    def sample_prior(
        self,
        shape_or_like: Sequence[int] | torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Sample the manifold's base distribution."""
        if isinstance(shape_or_like, torch.Tensor):
            shape = tuple(shape_or_like.shape)
            device = shape_or_like.device
            dtype = shape_or_like.dtype
        else:
            shape = tuple(shape_or_like)
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
        if len(shape) != 2:
            raise ValueError(
                "manifold samples must have shape (batch, features), "
                f"got {shape}"
            )
        samples = self.manifold.sample(shape[0], device=device, dtype=dtype)
        if samples.shape != shape:
            raise ValueError(
                f"manifold.sample returned shape {samples.shape}, expected {shape}"
            )
        return samples.to(device=device, dtype=dtype)

    def sample_time(
        self,
        reference: torch.Tensor,
        *,
        distribution: TimeDistribution = "uniform",
        t_min: float = 0.0,
        constant_time: float = 0.5,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one time per batch item, returned as shape ``(batch, 1)``."""
        t_max = self.total_time - self.time_eps
        if not 0 <= t_min < t_max:
            raise ValueError(
                "t_min must be non-negative and smaller than total_time - time_eps"
            )

        if t is not None:
            times = self.batch_time(t, reference)
        elif distribution == "constant":
            times = torch.full(
                (reference.shape[0], 1),
                constant_time,
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            unit = torch.rand(
                (reference.shape[0], 1),
                device=reference.device,
                dtype=reference.dtype,
            )
            if distribution == "quadratic":
                unit = unit.square()
            elif distribution != "uniform":
                raise ValueError(f"unknown time distribution: {distribution}")
            times = t_min + unit * (t_max - t_min)

        if torch.any(times < 0) or torch.any(times >= self.total_time):
            raise ValueError(f"training times must be in [0, {self.total_time})")
        return times

    def time_grid(self, reference: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Return an inclusive fixed grid from zero to ``total_time``."""
        return torch.linspace(
            0,
            self.total_time,
            n_steps + 1,
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def batch_time(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        """Convert scalar or per-sample time to shape ``(batch, 1)``."""
        time = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if time.ndim == 0:
            time = time.expand(x.shape[0]).unsqueeze(-1)
        elif time.ndim == 1:
            if time.shape[0] != x.shape[0]:
                raise ValueError(
                    f"time must contain {x.shape[0]} values, got {time.shape[0]}"
                )
            time = time.unsqueeze(-1)
        if time.shape != (x.shape[0], 1):
            raise ValueError(
                f"time must have shape ({x.shape[0]}, 1), got {tuple(time.shape)}"
            )
        return time

    @staticmethod
    def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch times for broadcasting over all feature dimensions."""
        return t.reshape(t.shape[0], *([1] * (x.ndim - 1)))

    def _conditional_state(
        self,
        x_0: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor | float,
    ) -> torch.Tensor:
        time = torch.as_tensor(t, device=x_data.device, dtype=x_data.dtype)
        if time.ndim == 1 and time.shape[0] == x_data.shape[0]:
            time = time.unsqueeze(-1)
        return self.manifold.geodesic(x_0, x_data, time / self.total_time)

    def _prepare_base(
        self,
        x_data: torch.Tensor,
        x_0: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self._validate_points(x_data, "x_data")
        if x_0 is None:
            return self.sample_prior(x_data)
        self._validate_pair(x_0, x_data)
        return x_0.to(device=x_data.device, dtype=x_data.dtype)

    @staticmethod
    def _validate_points(x: torch.Tensor, name: str) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if x.ndim != 2:
            raise ValueError(
                f"{name} must have shape (batch, features), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")

    @classmethod
    def _validate_pair(cls, x_0: torch.Tensor, x_data: torch.Tensor) -> None:
        cls._validate_points(x_0, "x_0")
        cls._validate_points(x_data, "x_data")
        if x_0.shape != x_data.shape:
            raise ValueError(
                "x_0 and x_data must have the same shape, "
                f"got {x_0.shape} and {x_data.shape}"
            )


__all__ = ["BaseFlowMatching", "FlowModel", "TimeDistribution"]
