"""Small fixed-grid ODE integrators.

The numerical layer knows nothing about losses, priors, neural networks or a
particular flow-matching parameterization.  Vector fields use the conventional
``f(t, x)`` ordering.  ``state_update`` and ``transport`` callbacks make the
same solvers usable in Euclidean space and on manifolds with a retraction or
exponential map.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import torch


VectorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StateUpdate = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TangentTransport = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _euclidean_update(x: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
    return x + increment


def _identity_transport(
    _source: torch.Tensor,
    _target: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    return tangent


class BaseODEIntegrator(ABC):
    """Base class for deterministic fixed-grid integrators."""

    @torch.inference_mode()
    def integrate(
        self,
        vector_field: VectorField,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        *,
        state_update: Optional[StateUpdate] = None,
        transport: Optional[TangentTransport] = None,
        return_trajectory: bool = False,
    ):
        """Integrate ``dx/dt = vector_field(t, x)`` over ``t_span``.

        ``t_span`` may be increasing or decreasing, which supports both
        generation and inversion.  When ``return_trajectory`` is true the
        result is ``(times, states)`` with ``states.shape[0] == len(times)``.
        """
        times = self._validate_inputs(x0, t_span)
        update = state_update or _euclidean_update
        tangent_transport = transport or _identity_transport

        x = x0
        states = [x.clone()] if return_trajectory else None
        for index in range(times.numel() - 1):
            t = times[index]
            dt = times[index + 1] - t
            x = self.step(
                vector_field,
                x,
                t,
                dt,
                state_update=update,
                transport=tangent_transport,
            )
            if states is not None:
                states.append(x.clone())

        if states is not None:
            return times, torch.stack(states, dim=0)
        return x

    @abstractmethod
    def step(
        self,
        vector_field: VectorField,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        *,
        state_update: StateUpdate = _euclidean_update,
        transport: TangentTransport = _identity_transport,
    ) -> torch.Tensor:
        """Advance one numerical step."""
        raise NotImplementedError

    @staticmethod
    def _validate_inputs(x0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        if not isinstance(x0, torch.Tensor) or not x0.is_floating_point():
            raise TypeError("x0 must be a floating-point torch.Tensor")
        times = torch.as_tensor(t_span, device=x0.device, dtype=x0.dtype)
        if times.ndim != 1 or times.numel() < 2:
            raise ValueError("t_span must be a 1-D tensor containing at least 2 times")
        deltas = times[1:] - times[:-1]
        if torch.any(deltas == 0):
            raise ValueError("adjacent values in t_span must be distinct")
        if not (torch.all(deltas > 0) or torch.all(deltas < 0)):
            raise ValueError("t_span must be strictly monotonic")
        return times

    @staticmethod
    def _evaluate(
        vector_field: VectorField,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tangent = vector_field(t, x)
        if tangent.shape != x.shape:
            raise ValueError(
                f"vector_field must return shape {x.shape}, got {tangent.shape}"
            )
        return tangent


class EulerIntegrator(BaseODEIntegrator):
    """First-order explicit Euler integration."""

    def step(
        self,
        vector_field: VectorField,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        *,
        state_update: StateUpdate = _euclidean_update,
        transport: TangentTransport = _identity_transport,
    ) -> torch.Tensor:
        del transport
        velocity = self._evaluate(vector_field, t, x)
        return state_update(x, dt * velocity)


class HeunIntegrator(BaseODEIntegrator):
    """Second-order explicit trapezoidal (improved Euler) integration."""

    def step(
        self,
        vector_field: VectorField,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        *,
        state_update: StateUpdate = _euclidean_update,
        transport: TangentTransport = _identity_transport,
    ) -> torch.Tensor:
        k1 = self._evaluate(vector_field, t, x)
        predictor = state_update(x, dt * k1)
        k2 = self._evaluate(vector_field, t + dt, predictor)
        k2_at_x = transport(predictor, x, k2)
        return state_update(x, 0.5 * dt * (k1 + k2_at_x))


class RK4Integrator(BaseODEIntegrator):
    """Classical fourth-order Runge--Kutta integration."""

    def step(
        self,
        vector_field: VectorField,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        *,
        state_update: StateUpdate = _euclidean_update,
        transport: TangentTransport = _identity_transport,
    ) -> torch.Tensor:
        half_dt = 0.5 * dt
        k1 = self._evaluate(vector_field, t, x)

        x2 = state_update(x, half_dt * k1)
        k2 = self._evaluate(vector_field, t + half_dt, x2)
        k2_at_x = transport(x2, x, k2)

        x3 = state_update(x, half_dt * k2_at_x)
        k3 = self._evaluate(vector_field, t + half_dt, x3)
        k3_at_x = transport(x3, x, k3)

        x4 = state_update(x, dt * k3_at_x)
        k4 = self._evaluate(vector_field, t + dt, x4)
        k4_at_x = transport(x4, x, k4)

        increment = dt * (k1 + 2 * k2_at_x + 2 * k3_at_x + k4_at_x) / 6
        return state_update(x, increment)


def get_integrator(
    integrator: str | BaseODEIntegrator,
) -> BaseODEIntegrator:
    """Resolve a configured integrator name or return an existing instance."""
    if isinstance(integrator, BaseODEIntegrator):
        return integrator
    if not isinstance(integrator, str):
        raise TypeError("integrator must be a name or BaseODEIntegrator instance")
    integrators = {
        "euler": EulerIntegrator,
        "heun": HeunIntegrator,
        "rk4": RK4Integrator,
    }
    try:
        return integrators[integrator.lower()]()
    except KeyError as exc:
        choices = ", ".join(integrators)
        raise ValueError(f"unknown integrator {integrator!r}; choose from {choices}") from exc


__all__ = [
    "BaseODEIntegrator",
    "EulerIntegrator",
    "HeunIntegrator",
    "RK4Integrator",
    "StateUpdate",
    "TangentTransport",
    "VectorField",
    "get_integrator",
]
