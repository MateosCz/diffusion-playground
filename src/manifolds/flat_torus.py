"""Flat-torus geometry in periodic coordinates."""

import math

import torch

from .base import BaseManifold


class FlatTorus(BaseManifold):
    """The flat torus ``R^dim / (period * Z)^dim``.

    Points are stored as coordinates in the half-open fundamental domain
    ``[center - period / 2, center + period / 2)``.  Tangent vectors use the
    same tensor representation as points.

    Args:
        dim: Dimension of the torus.
        period: Period of every coordinate.
        center: Center of the chosen fundamental domain.
        eps: Numerical tolerance inherited from :class:`BaseManifold`.
    """

    def __init__(
        self,
        dim: int = 2,
        period: float = 2 * math.pi,
        center: float = 0.0,
        eps: float = 1e-7,
    ):
        super().__init__(eps=eps)
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")

        self.dim = dim
        self.period = period
        self.center = center

    @property
    def lower(self) -> float:
        """Lower endpoint of the canonical fundamental domain."""
        return self.center - self.period / 2

    def wrap(self, samples: torch.Tensor) -> torch.Tensor:
        """Return the canonical representative of periodic coordinates."""
        return torch.remainder(samples - self.lower, self.period) + self.lower

    def unwrap(self, samples: torch.Tensor) -> torch.Tensor:
        """Map torus points to their canonical representative in ``R^dim``.

        A torus point has infinitely many Euclidean representatives.  This
        method chooses the one in this torus's fundamental domain.
        """
        return self.wrap(samples)

    def project_to_tangent(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Project ``v`` onto ``T_x M`` (the identity for a flat torus)."""
        return v

    def log_map(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return a shortest signed displacement from ``x0`` to ``x1``."""
        half_period = self.period / 2
        return torch.remainder(x1 - x0 + half_period, self.period) - half_period

    def exp_map(self, x0: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Move from ``x0`` along tangent vector ``v`` and wrap the result."""
        return self.wrap(x0 + v)

    @staticmethod
    def _broadcast_time(t, reference: torch.Tensor) -> torch.Tensor:
        """Turn scalar or batch-shaped time values into broadcastable values."""
        time = torch.as_tensor(t, device=reference.device, dtype=reference.dtype)
        while time.ndim < reference.ndim:
            time = time.unsqueeze(-1)
        return time

    def geodesic(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t,
    ) -> torch.Tensor:
        """Evaluate a constant-speed shortest geodesic at time ``t``.

        ``t=0`` returns ``x0`` and ``t=1`` returns the canonical
        representative of ``x1``.  Batched times such as shape ``(batch,)``
        are supported in addition to scalar times.
        """
        displacement = self.log_map(x0, x1)
        time = self._broadcast_time(t, displacement)
        return self.exp_map(x0, time * displacement)

    def geodesic_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t,
    ) -> torch.Tensor:
        """Return the constant tangent velocity of the shortest geodesic."""
        # Materialize ``t`` to catch invalid devices/dtypes consistently with
        # ``geodesic``.  The velocity itself is time independent.
        self._broadcast_time(t, x1 - x0)
        return self.log_map(x0, x1)

    def sample(self, batch_size: int, device="cpu") -> torch.Tensor:
        """Sample ``batch_size`` points uniformly from the torus."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError(
                f"batch_size must be an integer, got {type(batch_size).__name__}"
            )
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")

        samples = torch.rand(batch_size, self.dim, device=device)
        return samples * self.period + self.lower

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the geodesic distance under the flat Euclidean metric."""
        return torch.linalg.vector_norm(self.log_map(x, y), dim=-1)


class FlatTorus01(FlatTorus):
    """Represent a flat torus on the ``[0, 1)^D`` subspace.

    This manifold is isometric to a product of ``D`` one-dimensional circles.
    Points are represented by fractional coordinates, with coordinates that
    differ by an integer identifying the same point on the torus.

    Args:
        dim: Dimension of the torus.
        eps: Numerical tolerance inherited from :class:`BaseManifold`.
    """

    def __init__(self, dim: int = 2, eps: float = 1e-7):
        super().__init__(dim=dim, period=1.0, center=0.5, eps=eps)


# A descriptive alias used by some callers and by ``src.diffusion.rfm``.
FlatTorusManifold = FlatTorus


__all__ = ["FlatTorus", "FlatTorus01", "FlatTorusManifold"]
