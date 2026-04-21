from torch import nn
import torch
import src.scoreNNBlock as Block
from typing import Sequence, Union

class SimpleScoreMLP(nn.Module):
    def __init__(
        self,
        dim:int,
        hidden_dim: Union[Sequence[int], int],
        output_dim: int,
        **kwargs):
