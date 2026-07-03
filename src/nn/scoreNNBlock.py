from torch import nn
import torch
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for time-dependent score network.
    For each frequency w_i:
      emb[..., i]            = sin(2*pi*t*w_i)
      emb[..., i + half_dim] = cos(2*pi*t*w_i)
      accept time as (B,)
    return the embedding with shape (B, half_dim * 2)
    """
    def __init__(self, half_dim: int, max_period: float = 10000.0, use_2pi: bool = True):
        super().__init__()
        self.half_dim = half_dim
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
        # Accept time as (B,)
        t = t.to(dtype=self.freqs.dtype, device=self.freqs.device)
        angles = t * self.freqs
        if self.use_2pi:
            angles = angles * (2.0 * math.pi)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, half_dim * 2)
        return emb

class SinEmbedding(nn.Module):
    """
    Sinusoidal embedding of a (relative) coordinate vector, following the
    DiffCSP CSPNet design. Maps a tensor of shape (..., n_space) to
    (..., n_space * n_frequencies * 2) sin/cos features.

    Used to embed fractional-coordinate differences (and, via a projection,
    velocity differences) on the edges of a crystal graph.
    """
    def __init__(self, n_frequencies: int = 10, n_space: int = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.dim = n_space * n_frequencies * 2
        freqs = 2.0 * math.pi * torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_space)
        emb = x.unsqueeze(-1) * self.frequencies          # (..., n_space, n_frequencies)
        emb = emb.reshape(*x.shape[:-1], self.n_space * self.n_frequencies)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)    # (..., dim)
        return emb


class FourierEmbedding(nn.Module):
    """
    Random Fourier-feature embedding (e.g. for diffusion time). Maps an input
    of shape (B, in_features) to (B, out_features) via
    [sin(2*pi * x @ W), cos(2*pi * x @ W)] with fixed Gaussian weights W.
    """
    def __init__(self, in_features: int = 1, out_features: int = 128, scale: float = 1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, got {out_features}")
        self.in_features = in_features
        self.out_features = out_features
        half_dim = out_features // 2
        self.register_buffer("weight", torch.randn(in_features, half_dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,) or (B, in_features)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        x = x.to(dtype=self.weight.dtype, device=self.weight.device)
        proj = 2.0 * math.pi * (x @ self.weight)           # (B, half_dim)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # (B, out_features)


def sinusoidal_positional_embedding(token_sequence_size, half_dim, n=10000.0):


    T = token_sequence_size
    d = 2 * half_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, half_dim)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings


def sinusoidal_time_embedding(time: torch.Tensor, half_dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Sinusoidal time embedding.

    Args:
        time:       (B, 1) tensor of timesteps.
        half_dim:   number of frequency bands; output dim will be 2 * half_dim.
        max_period: controls the minimum frequency (largest period).

    Returns:
        (B, 2 * half_dim) tensor of sin/cos embeddings.
    """
    device = time.device
    dtype = torch.float32

    # (half_dim,)
    exponent = -math.log(max_period) * torch.arange(half_dim, device=device, dtype=dtype) / half_dim
    freqs = torch.exp(exponent)

    # (B, 1) * (1, half_dim) -> (B, half_dim)
    args = time.to(dtype) * freqs[None, :]

    # (B, 2 * half_dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb