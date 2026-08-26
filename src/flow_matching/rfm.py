"""Riemannian Flow Matching (RFM) with velocity regression."""

from collections.abc import Callable
from typing import Optional

import torch

from src.flow_matching.base import BaseFlowMatching, FlowModel, TimeDistribution
from src.manifolds.base import BaseManifold
from src.ode.integrators import BaseODEIntegrator, EulerIntegrator


class RiemannianFlowMatching(BaseFlowMatching):
    """Conditional flow matching along shortest Riemannian geodesics.

    The network directly predicts the tangent vector field.  Training follows
    the flow-matching convention ``(t, x_t, u_t)`` where

    ``u_t = Log_xt(x_data) / (total_time - t)``.
    """

    def __init__(
        self,
        manifold: BaseManifold,
        *,
        total_time: float = 1.0,
        time_eps: float = 1e-5,
        normalize_loss: bool = True,
        integrator: str | BaseODEIntegrator = "euler",
    ) -> None:
        super().__init__(
            manifold,
            total_time=total_time,
            time_eps=time_eps,
            integrator=integrator,
        )
        self.normalize_loss = normalize_loss

    def sample_training_pair(
        self,
        x_data: torch.Tensor,
        x_base: Optional[torch.Tensor] = None,
        *,
        t: Optional[torch.Tensor] = None,
        time_distribution: TimeDistribution = "uniform",
        t_min: float = 0.0,
        constant_time: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(t, x_t, conditional_velocity)`` for RFM training."""
        x_base = self._prepare_base(x_data, x_base)
        times = self.sample_time(
            x_data,
            distribution=time_distribution,
            t_min=t_min,
            constant_time=constant_time,
            t=t,
        )
        x_t = self._conditional_state(x_base, x_data, times)
        target = self.conditional_vector_field(times, x_t, x_data)
        return times, x_t, target

    def conditional_vector_field(
        self,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
        x_data: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form tangent target of the conditional geodesic path."""
        if x_t.shape != x_data.shape:
            raise ValueError(
                f"x_t and x_data must have the same shape, got {x_t.shape} "
                f"and {x_data.shape}"
            )
        times = self.batch_time(t, x_t)
        remaining = self.total_time - self.expand_time(times, x_t)
        if torch.any(remaining <= 0):
            raise ValueError("RFM target is undefined at or after total_time")
        tangent = self.manifold.log_map(x_t, x_data) / remaining
        return self.manifold.project_to_tangent(x_t, tangent)

    def loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean squared Riemannian error between two tangent fields."""
        del t
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have the same shape, "
                f"got {prediction.shape} and {target.shape}"
            )
        if x_t is None:
            x_t = target
        elif x_t.shape != target.shape:
            raise ValueError(
                f"x_t and target must have the same shape, got {x_t.shape} "
                f"and {target.shape}"
            )
        error = prediction - target
        result = self.manifold.squared_norm(x_t, error).mean()
        if self.normalize_loss:
            result = result / target.shape[-1]
        return result

    def vector_field(
        self,
        model: FlowModel,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a velocity-predicting model using ``model(t, x)``."""
        times = self.batch_time(t, x_t)
        velocity = model(times, x_t)
        if velocity.shape != x_t.shape:
            raise ValueError(
                f"model must return shape {x_t.shape}, got {velocity.shape}"
            )
        return self.manifold.project_to_tangent(x_t, velocity)

    def step(
        self,
        model: FlowModel,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        """Advance one Euler step of the learned flow."""
        scalar_t = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
        scalar_dt = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        return EulerIntegrator().step(
            lambda time, state: self.vector_field(model, time, state),
            x_t,
            scalar_t,
            scalar_dt,
            state_update=self.state_update,
        )

    # ------------------------------------------------------------------
    # Compatibility with the repository's former diffusion-style API.
    # New code should use sample_training_pair(), loss() and sample().
    # ------------------------------------------------------------------
    def sample_forward(
        self,
        x1: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        t_dist_kw: TimeDistribution = "uniform",
        constant_t: float = 0.5,
        return_time: bool = False,
        t_min: float = 0.0,
        ts: Optional[torch.Tensor] = None,
    ):
        t, x_t, target = self.sample_training_pair(
            x1,
            x0,
            t=ts,
            time_distribution=t_dist_kw,
            t_min=t_min,
            constant_time=constant_t,
        )
        if return_time:
            return x_t, target, t
        return x_t, target

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.loss(pred, target, x_t=x_t, t=t)

    @torch.inference_mode()
    def sample_backward(
        self,
        x0: torch.Tensor,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_steps: int = 100,
        sample_trajectory: bool = False,
    ):
        # The old callback convention was vector_field(x, t).
        def legacy_adapter(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return vector_field(x, self.batch_time(t, x))

        result = self.sample(
            legacy_adapter,
            x0,
            n_steps=n_steps,
            return_trajectory=sample_trajectory,
        )
        if sample_trajectory:
            _, states = result
            return states
        return result


RFM = RiemannianFlowMatching


__all__ = ["RFM", "RiemannianFlowMatching"]
