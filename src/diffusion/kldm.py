import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence, Callable, Tuple
from src.diffusion.sde import VPSDE, BaseSDE, BaseSDEIntegrator, EulerIntegrator, LinearSchedule
from src.distribution import WrappedNormalDistribution
from src.diffusion.tdm import TDMDiffusion, BaseDiffusion
from src.diffusion.continuous import ContinuousDiffusion
from torch_geometric.data import Data


class KLDM(nn.Module):  
    def __init__(
        self, 
        l_diffusion: ContinuousDiffusion, 
        f_diffusion: TDMDiffusion, 
        h_diffusion: Optional[BaseDiffusion] = None):
        super().__init__()
        self.l_diffusion = l_diffusion
        self.f_diffusion = f_diffusion
        self.h_diffusion = h_diffusion