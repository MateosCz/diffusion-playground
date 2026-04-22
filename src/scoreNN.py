from torch import nn
import torch
import src.scoreNNBlock as Block
from typing import Sequence, Union

class SimpleScoreMLP(nn.Module):
    def __init__(
        self,
        dim:int,
        x_lifting_dim: int,
        time_embedding_dim: int,
        hidden_dim: Union[Sequence[int], int],
        output_dim: int,
        **kwargs):
        super().__init__()
        self.dim = dim
        self.hidden_dim_list = hidden_dim
        self.output_dim = output_dim
        self.time_embedding_dim = time_embedding_dim
        self.time_embedding_layer = Block.SinusoidalTimeEmbedding(self.time_embedding_dim)
        self.score_net = nn.Sequential()
        self.x_lifting_dim = x_lifting_dim
        self.lifting_layer_x = nn.Linear(self.dim, self.x_lifting_dim)
        self.lifting_layer_hidden = nn.Linear(self.x_lifting_dim + self.time_embedding_dim, self.hidden_dim_list[0])
        for i in range(len(self.hidden_dim_list[:-1])):
            self.score_net.add_module(f"hidden_layer_{i}", nn.Linear(self.hidden_dim_list[i], self.hidden_dim_list[i+1]))
            self.score_net.add_module(f"relu_{i}", nn.ReLU())
        self.score_net.add_module(f"output_layer", nn.Linear(self.hidden_dim_list[-1], self.output_dim))


    """
    x: (batch_size, n_points, dim) or (n_points, dim), input data
    t: (batch_size, n_points, 1) or (batch_size, 1) or (batch_size,) or (), time tensor
    time embedding layer accept time as (), (B,), or (B,1), 
    so we only check the time with ndim 3
    """
    def forward(self,x: torch.Tensor, t: torch.Tensor):
        if t.ndim == 3:
            t = t[:,0,:]
        else:
            raise ValueError(f"Time dimension must be scalar, (B,), or (B,1) or (B, n_points, 1), got {t.shape}")



        t_emb = self.time_embedding_layer(t) # t_emb: (B, dim)
        # match the dimension of x and t_emb
        if t_emb.ndim != x.ndim:
            # if x is (n_points,dim)
            if x.ndim == 2:
                x = x[None,:,:]
            # if x is (batch_size, n_points, dim)
            elif x.ndim == 3:
                # match t_emb to x.shape
                t_emb = t_emb[:,None,:].expand(-1,x.shape[1],-1) # (B, n_points, dim)
            else:
                raise ValueError(f"Input data dimension must be (n_points,dim) or (batch_size, n_points, dim), got {x.shape}")
        # lift x first
        x = self.lifting_layer_x(x) # x: (B, n_points, x_lifting_dim)
        h_emb = torch.cat([x, t_emb], dim=-1) # (B, n_points, x_lifting_dim + time_embedding_dim)
        # lifting layer, lift the dimension of the input to the hidden_dim_list[0]
        h_emb = self.lifting_layer_hidden(h_emb) # (B, n_points, hidden_dim_list[0])
        return self.score_net(h_emb) # (B, n_points, output_dim)
