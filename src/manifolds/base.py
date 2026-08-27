"""Common interface for the geometries used by flow-matching models."""

from abc import ABC, abstractmethod

import torch


class BaseManifold(ABC):
    """Base class for all manifolds. 
    Reference:https://github.com/olgatticus/rg-vfm/blob/main/rvf/manifolds/base.py

    The implementation should provide:
    1. Basic operations: projection to tangent space, exponential and logarithmic maps
    2. Geodesic operations: geodesic paths and velocities
    3. Sampling: uniform sampling on the manifold
    4. Intrinsic/ambient representations for extrinsic flow models
    """
    
    def __init__(self, eps=1e-7):
        self.eps = eps
    
    @abstractmethod
    def wrap(self, samples: torch.Tensor) -> torch.Tensor:
        """Map coordinates to the manifold's canonical representation."""
        raise NotImplementedError
        
    @abstractmethod
    def unwrap(self, samples: torch.Tensor) -> torch.Tensor:
        """Map manifold points to canonical intrinsic coordinates."""
        raise NotImplementedError
        
    @abstractmethod
    def project_to_tangent(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Project vector v onto tangent space at point x."""
        raise NotImplementedError
        
    @abstractmethod
    def geodesic(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute geodesic between x0 and x1 at time t."""
        raise NotImplementedError
        
    @abstractmethod
    def geodesic_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute velocity of geodesic between x0 and x1 at time t."""
        raise NotImplementedError
        
    @abstractmethod
    def log_map(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Compute logarithmic map from x0 to x1."""
        raise NotImplementedError
        
    @abstractmethod
    def exp_map(self, x0: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute exponential map of v at x0."""
        raise NotImplementedError
        
    @abstractmethod
    def sample(
        self,
        batch_size: int,
        device="cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Sample points uniformly from manifold."""
        raise NotImplementedError 
    
    @abstractmethod
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance between ``x`` and ``y``."""
        raise NotImplementedError

    @abstractmethod
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Project the point outside the manifold to the manifold."""
        raise NotImplementedError

    def inner(
        self,
        x: torch.Tensor,
        tangent_a: torch.Tensor,
        tangent_b: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian inner product for an induced Euclidean metric."""
        del x
        return torch.sum(tangent_a * tangent_b, dim=-1)

    def squared_norm(
        self,
        x: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        """Squared norm of a tangent vector."""
        tangent = self.project_to_tangent(x, tangent)
        return self.inner(x, tangent, tangent)

    def parallel_transport(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        """Transport ``tangent`` from ``source`` to ``target``.

        Higher-order manifold integrators require this operation.  Manifolds
        without an implementation can still use Euler integration.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement parallel_transport"
        )

    @property
    @abstractmethod
    def intrinsic_dim(self) -> int:
        """Dimension of intrinsic coordinates used by this implementation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Dimension of the Euclidean space containing the embedding."""
        raise NotImplementedError

    @abstractmethod
    def to_ambient(self, points: torch.Tensor) -> torch.Tensor:
        """Embed intrinsic manifold coordinates in ambient Euclidean space."""
        raise NotImplementedError

    @abstractmethod
    def from_ambient(self, points: torch.Tensor) -> torch.Tensor:
        """Convert ambient points to canonical intrinsic coordinates."""
        raise NotImplementedError

    @abstractmethod
    def project_ambient(self, points: torch.Tensor) -> torch.Tensor:
        """Project arbitrary ambient points onto the embedded manifold."""
        raise NotImplementedError

    def sample_ambient(
        self,
        batch_size: int,
        device="cpu",
        dtype: torch.dtype | None = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample an isotropic Gaussian prior in the embedding space."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if scale <= 0:
            raise ValueError("scale must be positive")
        return scale * torch.randn(
            batch_size,
            self.ambient_dim,
            device=device,
            dtype=dtype,
        )

    def ambient_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance after projecting ambient points to the manifold."""
        return self.distance(self.from_ambient(x), self.from_ambient(y))


__all__ = ["BaseManifold"]
