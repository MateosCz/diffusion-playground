"""Time-conditioned MLP for RG-VFM endpoint prediction."""

from collections.abc import Sequence

import torch
from torch import nn

import src.nn.scoreNNBlock as Block


class RGVFMMLP(nn.Module):
    """Predict the RG-VFM endpoint ``x_T`` from ``(t, x_t)``.

    The network mirrors :class:`TDM_SimpleScoreMLP`: position and time
    features are lifted separately, time features are injected into every
    hidden layer, and SiLU activations are used throughout. Unlike the score
    network, it does not take a velocity input. Positions are represented only
    by their sine/cosine Fourier features so periodic coordinates remain
    continuous across the boundary.

    In flow matching, the endpoint x_T is usually x_1, and the total time is 1.0.
    """

    def __init__(
        self,
        dim: int,
        x_lifting_dim: int,
        time_embedding_half_dim: int,
        hidden_dim: Sequence[int] | int,
        output_dim: int,
        total_time: float = 1.0,
        time_embedding_scale: float = 1.0,
        position_fourier_bands: int = 1,
        position_period: float | Sequence[float] | torch.Tensor = 2 * torch.pi,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs


        # check the input arguments
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        if x_lifting_dim < 1:
            raise ValueError(
                f"x_lifting_dim must be positive, got {x_lifting_dim}"
            )
        if time_embedding_half_dim < 1:
            raise ValueError(
                "time_embedding_half_dim must be positive, "
                f"got {time_embedding_half_dim}"
            )
        if output_dim < 1:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if total_time <= 0:
            raise ValueError(f"total_time must be positive, got {total_time}")
        if position_fourier_bands < 1:
            raise ValueError(
                "position_fourier_bands must be positive, "
                f"got {position_fourier_bands}"
            )
        # end of checking the input arguments


        # check the hidden_dim
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        self.hidden_dim_list = list(hidden_dim)
        if not self.hidden_dim_list or any(width < 1 for width in self.hidden_dim_list):
            raise ValueError("hidden_dim must contain at least one positive width")
        # end of checking the hidden_dim

        self.dim = dim
        self.output_dim = output_dim
        self.x_lifting_dim = x_lifting_dim
        self.time_embedding_half_dim = time_embedding_half_dim
        self.time_embedding_dim = 2 * time_embedding_half_dim
        self.total_time = float(total_time)
        self.time_embedding_scale = float(time_embedding_scale)
        self.position_fourier_bands = position_fourier_bands

        position_period_tensor = torch.as_tensor(
            position_period,
            dtype=torch.float32,
        )
        if position_period_tensor.ndim == 0:
            position_period_tensor = position_period_tensor.repeat(dim)
        if position_period_tensor.shape != (dim,):
            raise ValueError(
                "position_period must be a scalar or have shape "
                f"({dim},), got {tuple(position_period_tensor.shape)}"
            )
        if not torch.isfinite(position_period_tensor).all():
            raise ValueError("position_period must contain only finite values")
        if torch.any(position_period_tensor <= 0):
            raise ValueError("position_period must contain only positive values")
        self.register_buffer(
            "position_period",
            position_period_tensor.clone(),
        )

        position_embedding_dim = dim * 2 * position_fourier_bands
        self.register_buffer(
            "position_frequencies",
            torch.arange(1, position_fourier_bands + 1, dtype=torch.float32),
        )

        self.lifting_layer_x = nn.Sequential(
            nn.Linear(position_embedding_dim, x_lifting_dim),
            nn.SiLU(),
            nn.Linear(x_lifting_dim, x_lifting_dim),
        )
        self.lifting_layer_t = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        self.norm = nn.LayerNorm(x_lifting_dim + self.time_embedding_dim)
        self.lifting_layer_hidden = nn.Linear(
            x_lifting_dim + self.time_embedding_dim,
            self.hidden_dim_list[0],
        )

        self.endpoint_net = nn.ModuleList(
            nn.Linear(current_width + self.time_embedding_dim, next_width)
            for current_width, next_width in zip(
                self.hidden_dim_list[:-1],
                self.hidden_dim_list[1:],
            )
        )
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(self.hidden_dim_list[-1], output_dim)

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Return the predicted terminal state with shape ``(batch, output_dim)``."""
        if x_t.ndim != 2 or x_t.shape[-1] != self.dim:
            raise ValueError(
                f"x_t must have shape (batch, {self.dim}), got {tuple(x_t.shape)}"
            )
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape != (x_t.shape[0], 1):
            raise ValueError(
                f"t must have shape ({x_t.shape[0]}, 1), got {tuple(t.shape)}"
            )

        frequencies = self.position_frequencies.to(dtype=x_t.dtype)
        period = self.position_period.to(dtype=x_t.dtype)
        normalized_x = x_t / period
        x_frequencies = (
            2 * torch.pi * normalized_x.unsqueeze(-1) * frequencies
        )
        x_embedding = torch.cat(
            [
                torch.sin(x_frequencies).flatten(start_dim=-2),
                torch.cos(x_frequencies).flatten(start_dim=-2),
            ],
            dim=-1,
        )

        normalized_t = t / self.total_time
        t_embedding = Block.sinusoidal_time_embedding(
            normalized_t * self.time_embedding_scale,
            self.time_embedding_half_dim,
        ).to(dtype=x_t.dtype)
        h_t = self.lifting_layer_t(t_embedding)
        h_x = self.lifting_layer_x(x_embedding)

        hidden = self.norm(torch.cat([h_x, h_t], dim=-1))
        hidden = self.lifting_layer_hidden(hidden)
        for hidden_layer in self.endpoint_net:
            hidden = hidden_layer(torch.cat([hidden, h_t], dim=-1))
            hidden = self.activation(hidden)
        return self.output_layer(hidden)


__all__ = ["RGVFMMLP"]
