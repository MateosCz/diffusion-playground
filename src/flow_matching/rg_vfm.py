"""Riemannian Gaussian Variational Flow Matching (RG-VFM).

RG-VFM predicts the terminal state ``x_T`` of a conditional probability path.
At sampling time the ``x_T`` prediction is converted into a vector field.
Intrinsic support uses a logarithmic map,

``v_t(x_t) = Log_{x_t}(pred_x_T) / (total_time - t)``.

Extrinsic support uses the analogous Euclidean displacement in the manifold's
declared ambient space.
"""

from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch

from src.flow_matching.base import BaseFlowMatching, FlowModel, TimeDistribution
from src.manifolds.base import BaseManifold
from src.ode.integrators import BaseODEIntegrator, EulerIntegrator


class RiemannianGaussianVariationalFlowMatching(BaseFlowMatching):
    """RG-VFM with intrinsic or ambient-space terminal-state regression.

    ``support="intrinsic"`` uses geodesics, tangent vector fields and manifold
    priors.  ``support="extrinsic"`` embeds data into the manifold's ambient
    Euclidean space, samples a Gaussian base there and uses linear conditional
    paths.  In both cases the objective remains squared geodesic distance.
    """

    def __init__(
        self,
        manifold: BaseManifold,
        *,
        total_time: float = 1.0,
        time_eps: float = 1e-5,
        noise_scale: float = 0.0,
        max_velocity_scale: Optional[float] = 5.0,
        normalize_loss: bool = False,
        support: Literal["intrinsic", "extrinsic"] = "intrinsic",
        intrinsic_prior_std: float = 1.0,
        ambient_prior_scale: float = 1.0,
        integrator: str | BaseODEIntegrator = "euler",
    ) -> None:
        super().__init__(
            manifold,
            total_time=total_time,
            time_eps=time_eps,
            integrator=integrator,
        )
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
        if max_velocity_scale is not None and max_velocity_scale <= 0:
            raise ValueError(
                "max_velocity_scale must be positive or None, "
                f"got {max_velocity_scale}"
            )
        if support not in ("intrinsic", "extrinsic"):
            raise ValueError(
                f"support must be 'intrinsic' or 'extrinsic', got {support!r}"
            )
        if intrinsic_prior_std <= 0:
            raise ValueError("intrinsic_prior_std must be positive")
        if ambient_prior_scale <= 0:
            raise ValueError("ambient_prior_scale must be positive")
        self.noise_scale = float(noise_scale)
        self.max_velocity_scale = max_velocity_scale
        self.normalize_loss = normalize_loss
        self.support = support
        self.intrinsic_prior_std = float(intrinsic_prior_std)
        self.ambient_prior_scale = float(ambient_prior_scale)

    @property
    def model_dim(self) -> int:
        """Feature dimension seen by the ``x_T``-predicting model."""
        if self.support == "extrinsic":
            return self.manifold.ambient_dim
        return self.manifold.intrinsic_dim

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
        """Return ``(t, x_t, x_T)`` in the configured representation."""
        x_0, x_T = self._prepare_training_pair(x_data, x_0)
        times = self.sample_time(
            x_T,
            distribution=time_distribution,
            t_min=t_min,
            constant_time=constant_time,
            t=t,
        )
        x_t = self._conditional_state(x_0, x_T, times)
        return times, self._perturb(x_t), x_T

    def loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Time-weighted squared geodesic distance to the target ``x_T``.

        Each sample is weighted by ``1 / (total_time - t)^2`` before the
        batch reduction.  Omitting ``t`` is equivalent to evaluating the
        objective at ``t=0``.
        """
        del x_t
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have the same shape, "
                f"got {prediction.shape} and {target.shape}"
            )
        if self.support == "extrinsic":
            distance = self.manifold.ambient_distance(prediction, target) # geodesic distance after projecting the ambient space to the manifold
        else:
            distance = self.manifold.distance(target, prediction) # geodesic distance in the manifold
        if t is None:
            times = torch.zeros(
                (prediction.shape[0], 1),
                device=prediction.device,
                dtype=prediction.dtype,
            )
        else:
            times = self.batch_time(t, prediction)
        remaining = self.total_time - times.squeeze(-1)
        if torch.any(remaining <= 0):
            raise ValueError("RG-VFM loss is undefined at or after total_time")
        weight = remaining.pow(-2)
        result = (weight * distance.pow(2)).mean()
        if self.normalize_loss:
            result = result / self.manifold.intrinsic_dim
        return result

    def x_T_to_vector_field(
        self,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a predicted terminal state ``x_T`` into a tangent velocity."""
        self._validate_model_state(x_t, "x_t")
        if x_T.shape != x_t.shape:
            raise ValueError(
                "x_T and x_t must have the same shape, "
                f"got {x_T.shape} and {x_t.shape}"
            )
        times = self.batch_time(t, x_t)
        remaining = self.total_time - self.expand_time(times, x_t)
        if torch.any(remaining < 0):
            raise ValueError("RG-VFM velocity is undefined after total_time")
        if self.max_velocity_scale is None and torch.any(remaining == 0):
            raise ValueError(
                "the exact RG-VFM velocity is undefined at total_time; "
                "use Euler integration or configure max_velocity_scale"
            )
        scale = remaining.reciprocal()
        if self.max_velocity_scale is not None:
            scale = torch.clamp(scale,min=0, max=self.max_velocity_scale)
        if self.support == "extrinsic":
            return (x_T - x_t) * scale
        tangent = self.manifold.log_map(x_t, x_T) * scale
        return self.manifold.project_to_tangent(x_t, tangent)

    def vector_field(
        self,
        model: FlowModel,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate an ``x_T`` model and construct its velocity at ``x_t``."""
        times = self.batch_time(t, x_t)
        pred_x_T = model(times, x_t)
        if pred_x_T.shape != x_t.shape:
            raise ValueError(
                f"model must return shape {x_t.shape}, got {pred_x_T.shape}"
            )
        return self.x_T_to_vector_field(times, x_t, pred_x_T)

    def step(
        self,
        model: FlowModel,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        """Advance one Euler step using an ``x_T``-predicting model."""
        scalar_t = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
        scalar_dt = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        return EulerIntegrator().step(
            lambda time, state: self.vector_field(model, time, state),
            x_t,
            scalar_t,
            scalar_dt,
            state_update=self.state_update,
        )

    def state_update(
        self,
        x: torch.Tensor,
        increment: torch.Tensor,
    ) -> torch.Tensor:
        """Apply increments in intrinsic or ambient coordinates."""
        if self.support == "extrinsic":
            return x + increment
        return self.manifold.exp_map(x, increment)

    def tangent_transport(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        """Ambient transport is identity; intrinsic transport is geometric."""
        if self.support == "extrinsic":
            return tangent
        return self.manifold.parallel_transport(source, target, tangent)

    def sample_path(
        self,
        x_0: torch.Tensor,
        x_data: torch.Tensor,
        *,
        t: Optional[torch.Tensor | float] = None,
        n_steps: int = 100,
    ):
        """Evaluate a conditional path in intrinsic or ambient coordinates."""
        x_T = self.to_model_space(x_data)
        self._validate_pair(x_0, x_T)
        if t is not None:
            return self._conditional_state(x_0, x_T, t)
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        times = self.time_grid(x_T, n_steps)
        states = torch.stack(
            [self._conditional_state(x_0, x_T, time) for time in times],
            dim=0,
        )
        return times, states

    def sample_prior(
        self,
        shape_or_like: Sequence[int] | torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Sample a wrapped normal or an ambient Euclidean Gaussian prior."""
        if self.support == "intrinsic":
            if isinstance(shape_or_like, torch.Tensor):
                self._validate_points(shape_or_like, "shape_or_like")
                shape = tuple(shape_or_like.shape)
                device = shape_or_like.device
                dtype = shape_or_like.dtype
            else:
                shape = tuple(shape_or_like)
                device = device or torch.device("cpu")
                dtype = dtype or torch.get_default_dtype()
            if len(shape) != 2:
                raise ValueError("shape must be (batch, features)")
            if shape[1] != self.manifold.intrinsic_dim:
                raise ValueError(
                    "feature dimension must match intrinsic space, "
                    f"got {shape[1]}"
                )
            gaussian = self.intrinsic_prior_std * torch.randn(
                shape,
                device=device,
                dtype=dtype,
            )
            return self.manifold.wrap(gaussian)
        if isinstance(shape_or_like, torch.Tensor):
            self._validate_points(shape_or_like, "shape_or_like")
            if shape_or_like.shape[-1] not in (
                self.manifold.intrinsic_dim,
                self.manifold.ambient_dim,
            ):
                raise ValueError(
                    "shape_or_like feature dimension must match intrinsic or "
                    f"ambient space, got {shape_or_like.shape[-1]}"
                )
            batch_size = shape_or_like.shape[0]
            device = shape_or_like.device
            dtype = shape_or_like.dtype
        else:
            shape = tuple(shape_or_like)
            if len(shape) != 2:
                raise ValueError("shape must be (batch, features)")
            if shape[1] not in (
                self.manifold.intrinsic_dim,
                self.manifold.ambient_dim,
            ):
                raise ValueError(
                    "feature dimension must match intrinsic or ambient space, "
                    f"got {shape[1]}"
                )
            batch_size = shape[0]
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
        return self.manifold.sample_ambient(
            batch_size,
            device=device,
            dtype=dtype,
            scale=self.ambient_prior_scale,
        )

    def to_model_space(self, x_data: torch.Tensor) -> torch.Tensor:
        """Convert intrinsic data to the representation consumed by the model."""
        self._validate_points(x_data, "x_data")
        if self.support == "intrinsic":
            if x_data.shape[-1] != self.manifold.intrinsic_dim:
                raise ValueError(
                    f"intrinsic data must have {self.manifold.intrinsic_dim} "
                    f"features, got {x_data.shape[-1]}"
                )
            return self.manifold.wrap(x_data)
        if x_data.shape[-1] == self.manifold.intrinsic_dim:
            return self.manifold.to_ambient(x_data)
        if x_data.shape[-1] == self.manifold.ambient_dim:
            return self.manifold.project_ambient(x_data)
        raise ValueError(
            "extrinsic data must use either intrinsic dimension "
            f"{self.manifold.intrinsic_dim} or ambient dimension "
            f"{self.manifold.ambient_dim}, got {x_data.shape[-1]}"
        )

    def to_intrinsic(self, model_state: torch.Tensor) -> torch.Tensor:
        """Decode a model-space state to canonical intrinsic coordinates."""
        self._validate_model_state(model_state, "model_state")
        if self.support == "intrinsic":
            return self.manifold.wrap(model_state)
        return self.manifold.from_ambient(model_state)

    def project_model_state(self, model_state: torch.Tensor) -> torch.Tensor:
        """Project a generated state onto its valid model-space support."""
        self._validate_model_state(model_state, "model_state")
        if self.support == "intrinsic":
            return self.manifold.wrap(model_state)
        return self.manifold.project_ambient(model_state)

    def _validate_model_state(self, state: torch.Tensor, name: str) -> None:
        self._validate_points(state, name)
        if state.shape[-1] != self.model_dim:
            raise ValueError(
                f"{name} must have model dimension {self.model_dim}, "
                f"got {state.shape[-1]}"
            )

    def _prepare_training_pair(
        self,
        x_data: torch.Tensor,
        x_0: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_T = self.to_model_space(x_data)
        if x_0 is None:
            x_0 = self.sample_prior(x_T)
        else:
            self._validate_points(x_0, "x_0")
            x_0 = x_0.to(device=x_T.device, dtype=x_T.dtype)
        self._validate_pair(x_0, x_T)
        return x_0, x_T

    def _conditional_state(
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor | float,
    ) -> torch.Tensor:
        if self.support == "intrinsic":
            return super()._conditional_state(x_0, x_T, t)
        time = torch.as_tensor(t, device=x_T.device, dtype=x_T.dtype)
        while time.ndim < x_T.ndim:
            time = time.unsqueeze(-1)
        return x_0 + (time / self.total_time) * (x_T - x_0)

    def _perturb(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.noise_scale == 0:
            return x_t
        if self.support == "extrinsic":
            return x_t + self.noise_scale * torch.randn_like(x_t)
        noise = self.manifold.project_to_tangent(x_t, torch.randn_like(x_t))
        return self.manifold.exp_map(x_t, self.noise_scale * noise)

    # Backward-compatible names from the endpoint-parameterized API. Keep the
    # old keyword name as well as the old method names.
    def endpoint_to_vector_field(
        self,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
        endpoint: torch.Tensor,
    ) -> torch.Tensor:
        return self.x_T_to_vector_field(t, x_t, endpoint)

    endpoint_to_velocity = endpoint_to_vector_field

    # ------------------------------------------------------------------
    # Compatibility with the earlier diffusion-style API.
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
        sample_trajectory: bool = False,
        n_steps: int = 100,
    ):
        if sample_trajectory:
            if ts is not None:
                raise ValueError("ts cannot be combined with sample_trajectory=True")
            x0, x_T = self._prepare_training_pair(x1, x0)
            times, states = self.sample_path(x0, x_T, n_steps=n_steps)
            states = self._perturb(states)
            targets = x_T.unsqueeze(0).expand_as(states)
            if return_time:
                return states, targets, times
            return states, targets

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
    ) -> torch.Tensor:
        return self.loss(pred, target, t=t)

    def sample_backward_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        pred: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        velocity = self.x_T_to_vector_field(t, x_t, pred)
        step_size = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        return self.state_update(x_t, step_size * velocity)

    backward_step = sample_backward_step

    @torch.inference_mode()
    def sample_backward(
        self,
        x0: torch.Tensor,
        endpoint_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_steps: int = 100,
        sample_trajectory: bool = False,
        return_time: bool = False,
    ):
        # The old callback convention was endpoint_model(x, t).
        def legacy_adapter(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return endpoint_model(x, self.batch_time(t, x))

        result = self.sample(
            legacy_adapter,
            x0,
            n_steps=n_steps,
            return_trajectory=sample_trajectory,
        )
        if sample_trajectory:
            times, states = result
            if return_time:
                return states, times
            return states
        if return_time:
            final_time = torch.full(
                (x0.shape[0], 1),
                self.total_time,
                device=x0.device,
                dtype=x0.dtype,
            )
            return result, final_time
        return result


RGVFM = RiemannianGaussianVariationalFlowMatching


__all__ = ["RGVFM", "RiemannianGaussianVariationalFlowMatching"]
