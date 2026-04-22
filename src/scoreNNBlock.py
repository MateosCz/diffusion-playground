from torch import nn
import torch
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for time-dependent score network.
    For each frequency w_i:
      emb[..., i]            = sin(2*pi*t*w_i)
      emb[..., i + half_dim] = cos(2*pi*t*w_i)
    return the embedding with shape (B, dim)
    """
    def __init__(self, dim: int, max_period: float = 10000.0, use_2pi: bool = True):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        if dim < 2:
            raise ValueError("time embedding dim must be >= 2")
        self.dim = dim
        self.half_dim = dim // 2
        self.max_period = max_period
        self.use_2pi = use_2pi
        # Handle dim=2 (half_dim=1) safely.
        if self.half_dim == 1:
            freqs = torch.ones(1, dtype=torch.float32)
        else:
            log_scale = math.log(max_period) / (self.half_dim - 1)
            freqs = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -log_scale)
        self.register_buffer("freqs", freqs)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Accept scalar (), vector (B,), or column (B,1)
        if t.ndim == 0:
            t = t[None]
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        elif t.ndim != 1:
            raise ValueError(f"Expected t with shape (), (B,), or (B,1), got {tuple(t.shape)}")
        t = t.to(dtype=self.freqs.dtype, device=self.freqs.device)
        angles = t[:, None] * self.freqs[None, :]
        if self.use_2pi:
            angles = angles * (2.0 * math.pi)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
        return emb

    