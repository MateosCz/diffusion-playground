"""Two-time endpoint predictors for intrinsic and extrinsic flow maps.

The networks return pi_{s,t}(x_s); training and sampling construct X_{s,t}.
"""

import torch

from src.nn.rg_vfm_mlp import RGVFMMLP, EX_RGVFMMLP


class InFlowMapNN(RGVFMMLP):
    """Predict the intrinsic endpoint pi_{s,t}(x_s), conditioned on s and t."""

    def __init__(
        self, *args,
        with_sincos_position=True,
        position_fourier_bands=8,
        residual_position_scale=0.01,
        **kwargs,
    ):
        """Default to periodic features and a 0.01 residual scale.

        The position period comes from manifold.period.
        """
        kwargs.update(time_input_dim=2, with_residual_position=True)
        super().__init__(
            *args,
            with_sincos_position=with_sincos_position,
            residual_position_scale=residual_position_scale,
            position_fourier_bands=position_fourier_bands,
            **kwargs,
        )

    def forward(self, x_s, s, t):
        if x_s.ndim != 2 or x_s.shape[-1] != self.dim:
            raise ValueError(f"x_s must have shape (batch, {self.dim})")
        s = s.unsqueeze(-1) if s.ndim == 1 else s
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        expected = (x_s.shape[0], 1)
        if s.shape != expected or t.shape != expected:
            raise ValueError("s and t must each have shape (B,) or (B, 1)")

        s = s.to(device=x_s.device, dtype=x_s.dtype)
        t = t.to(device=x_s.device, dtype=x_s.dtype)
        times = torch.cat([s, t], dim=-1)
        return super().forward(times, x_s)


class ExFlowMapNN(EX_RGVFMMLP):
    """Predict the ambient endpoint pi_{s,t}(x_s), conditioned on s and t."""

    def __init__(
        self, *args,
        with_sincos_position=False,
        residual_position_scale=0.01,
        **kwargs,
    ):
        """Default to raw ambient coordinates and a 0.01 residual scale.

        Endpoint predictions remain unprojected in ambient coordinates.
        """
        kwargs.update(
            time_input_dim=2, with_residual_position=True, project_to_manifold=False
        )
        super().__init__(
            *args,
            with_sincos_position=with_sincos_position,
            residual_position_scale=residual_position_scale,
            **kwargs,
        )

    def forward(self, x_s, s, t):
        if x_s.ndim != 2 or x_s.shape[-1] != self.dim:
            raise ValueError(f"x_s must have shape (batch, {self.dim})")
        s = s.unsqueeze(-1) if s.ndim == 1 else s
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        expected = (x_s.shape[0], 1)
        if s.shape != expected or t.shape != expected:
            raise ValueError("s and t must each have shape (B,) or (B, 1)")

        s = s.to(device=x_s.device, dtype=x_s.dtype)
        t = t.to(device=x_s.device, dtype=x_s.dtype)
        times = torch.cat([s, t], dim=-1)
        return super().forward(times, x_s)


# Keep existing imports working.
FracFlowMapNN = InFlowMapNN
EX_FracFlowMapNN = ExFlowMapNN

__all__ = ["InFlowMapNN", "ExFlowMapNN", "FracFlowMapNN", "EX_FracFlowMapNN"]
