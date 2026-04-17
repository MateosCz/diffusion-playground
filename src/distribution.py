import torch
from torch import nn
from typing import Optional, Literal
from abc import ABC, abstractmethod

class BaseDistribution(ABC,nn.Module):
    @abstractmethod
    def sample(self):
        raise NotImplementedError
    
    @abstractmethod
    def sample_n(self,n):
        raise NotImplementedError

    @abstractmethod
    def sample_like(self, shape_like: torch.Tensor):
        raise NotImplementedError


    @abstractmethod
    def sample_shape(self, shape: torch.Tensor):
        raise NotImplementedError

class WrappedNormalDistribution(BaseDistribution):
    def __init__(self, sigma, trunc_n):
        self.sigma = sigma
        self.trunc_n = trunc_n

    # def sample(self):


