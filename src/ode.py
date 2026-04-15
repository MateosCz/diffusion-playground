"""
Customized ODE module,
with ODE class and Integrators
or people can simply use the scipy ode class for ode integration.
"""


import torch
import torch.nn as nn
from collections.abc import Callable
from abc import ABC, abstractmethod
from typing import Optional



class ODE(nn.Module):
    """
    ODE class
    dx = f(x,t)dt
    constructor receives a drift_fn attribute
    drift_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor
    jacobian(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor
    """
    def __init__(
        self, 
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        jacobian: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]=None):
        self.drift_fn = drift_fn
        self.jacobian = jacobian

