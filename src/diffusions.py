"""
diffusion models

"""
from torch import nn
import torch
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence
from src.sde import BaseSDE, BaseSDEIntegrator


"""
Base class of diffusions, including the 
"""
class BaseDiffusion(ABC, nn.Module):
    
    """
    the loss function of DSM, given predicted score, target result and time t.
    """
    @abstractmethod
    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor
    ):
        raise NotImplementedError


    """
    inference step throw the time reversed diffusion 
    """
    @torch.inference_mode()
    def reverse_step(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            pred: torch.Tensor,
            dt: torch.Tensor,
            **_,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def sample_prior(self, index: torch.Tensor):
        raise NotImplementedError

    
"""
TDM (Trivialized Diffusion Model) used for doing diffusions on fractional coordinate data.
Input data should already be wrapped to LieTorus data (data in SO2^n).
Constructor accepts 2 parameters: the backbone sde and the dimension of input data.
"""

class TDMDiffusion(BaseDiffusion):
    def __init__(self, sde: BaseSDE, dim: Sequence[int] | int, integrator: BaseSDEIntegrator):
        self.sde = sde
        self.dim = dim
        self.integrator = integrator





