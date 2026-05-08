from torch import nn
import torch
import src.scoreNNBlock as Block
from typing import Sequence, Union

class TDM_SimpleScoreMLP(nn.Module):
    def __init__(
        self,
        dim:int,
        x_lifting_dim: int,
        time_embedding_half_dim: int,
        hidden_dim: Union[Sequence[int], int],
        output_dim: int,
        total_time: float = 2.0,
        time_embedding_scale: float = 1.0,
        position_fourier_bands: int = 1,
        with_sincos_position: bool = False,
        only_sincos_position: bool = False,
        **kwargs):
        super().__init__()
        self.dim = dim
        self.hidden_dim_list = hidden_dim
        self.output_dim = output_dim
        self.time_embedding_half_dim = time_embedding_half_dim
        self.total_time = total_time
        self.time_embedding_scale = time_embedding_scale
        self.position_fourier_bands = position_fourier_bands
        # self.time_embedding_layer = Block.SinusoidalTimeEmbedding(self.time_embedding_dim)
        self.with_sincos_position = with_sincos_position
        self.only_sincos_position = only_sincos_position
        self.score_net = nn.Sequential()
        self.x_lifting_dim = x_lifting_dim
        self.v_dim = self.dim
        if self.with_sincos_position:
            sincos_dim = self.dim * 2 * self.position_fourier_bands
            if self.only_sincos_position:
                self.dim = sincos_dim # only use sin cos position dimension to the input data
            else:   
                self.dim = self.dim + sincos_dim # add sin cos position dimension to the input data
        self.dim = self.dim + self.v_dim # add vt dimension to the input data
        
        self.lifting_layer_x = nn.Sequential(
            nn.Linear(self.dim, self.x_lifting_dim),
            nn.SiLU(),
            nn.Linear(self.x_lifting_dim, self.x_lifting_dim),
        )

        self.lifting_layer_t = nn.Sequential(
            nn.Linear(2 * self.time_embedding_half_dim, 2 * self.time_embedding_half_dim),
            nn.SiLU(),
            nn.Linear(2 * self.time_embedding_half_dim, 2 * self.time_embedding_half_dim),
        )

        self.norm = nn.LayerNorm(2 * self.time_embedding_half_dim)
    
        self.lifting_layer_hidden = nn.Linear(2 * self.time_embedding_half_dim, self.hidden_dim_list[0])
        
        for i in range(len(self.hidden_dim_list[:-1])):
            self.score_net.add_module(f"hidden_layer_{i}", nn.Linear(self.hidden_dim_list[i], self.hidden_dim_list[i+1]))
            self.score_net.add_module(f"leaky_relu_{i}", nn.SiLU())
        self.score_net.add_module(f"output_layer", nn.Linear(self.hidden_dim_list[-1], self.output_dim))




    """
    x: (batch_size, dim) , input data
    vt: (batch_size, dim), velocity tensor at time t,
    vt should have the same shape as x
    t: (batch_size, 1), time tensor easy to broadcast with x and vt
    return the score of the input data, shape (batch_size, output_dim)
    """
    def forward(self,x: torch.Tensor, vt: torch.Tensor, t: torch.Tensor):
        # check if the shape of x and vt are the same
        if x.shape != vt.shape:
            raise ValueError(f"Input data and velocity tensor dimension must be the same, got {x.shape} and {vt.shape}")
        # first check if the sin cos position is needed
        if self.with_sincos_position:
            frequencies = torch.arange(
                1,
                self.position_fourier_bands + 1,
                device=x.device,
                dtype=x.dtype,
            )
            x_freq = x.unsqueeze(-1) * frequencies
            sincos_x = torch.cat(
                [
                    torch.sin(x_freq).flatten(start_dim=-2),
                    torch.cos(x_freq).flatten(start_dim=-2),
                ],
                dim=-1,
            )
            if self.only_sincos_position:
                x = sincos_x
            else:
                x = torch.cat([x, sincos_x], dim=-1)
        x = torch.cat([x, vt], dim=-1)
        t_norm = t/self.total_time
        t_emb = Block.sinusoidal_time_embedding(t_norm * self.time_embedding_scale, self.time_embedding_half_dim)
        h_t = self.lifting_layer_t(t_emb)
        # t_emb = self.time_embedding_layer(t) # t_emb: (B, dim)
        # lift x first
        h_x = self.lifting_layer_x(x) # x: (B, x_lifting_dim)
        # h_emb = torch.cat([x, h_t], dim=-1) # (B, x_lifting_dim + time_embedding_dim)
        h_emb = self.norm(h_x + h_t)

        # lifting layer, lift the dimension of the input to the hidden_dim_list[0]
        h_emb = self.lifting_layer_hidden(h_emb) # (B, hidden_dim_list[0])
        h_out = self.score_net(h_emb) # (B, output_dim)
        return h_out

        
