"""Time-conditioned periodic MLP for Riemannian velocity prediction."""

import torch

from src.nn.rg_vfm_mlp import RGVFMMLP


class RFMMLP(RGVFMMLP):
    """Predict a tangent vector from ``(t, x_t)`` on a periodic manifold.

    The architecture and Fourier position features intentionally match
    :class:`RGVFMMLP`, but its output is a tangent vector and must therefore
    not be wrapped into the point-coordinate fundamental domain.
    """

    def _format_output(
        self,
        raw_output: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.manifold.project_to_tangent(x_t, raw_output)


__all__ = ["RFMMLP"]
