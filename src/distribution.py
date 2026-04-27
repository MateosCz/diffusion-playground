import torch
from torch import nn
from typing import Optional, Literal
from abc import ABC, abstractmethod

from src.data import wrap_angle, wrap_pos
import math

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


class WrappedNormalDistribution():
    def __init__(self, mu, sigma, trunc_n):
        self.mu = mu
        self.sigma = sigma
        self.trunc_n = trunc_n

    # given the value x, return the score of the distribution p(x)
    # because mu and sigma are t related, so the score is also t related
    def score(self, x, T:float = 2 * torch.pi):
        C_2 = 0
        C = 0
        for k in range(-self.trunc_n, self.trunc_n+1):
            C_component = torch.exp((-(x - self.mu + k * T)**2)/(2*(self.sigma**2)))
            C2_component = (x - self.mu + k *T) / (self.sigma**2)
            C_2 += C_component * C2_component
            C += C_component
        
        return C_2 / C
            



def sigma_norm(sigma: torch.Tensor, T: float = 2 * torch.pi, N: int = 10, sn: int = 20000):
    
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = wrap_pos(x_sample, x_range=(2.0 * math.pi) / T)
    WN = WrappedNormalDistribution(mu = torch.zeros_like(x_sample), sigma = sigma, trunc_n = N)
    normal_ = WN.score(
        x_sample,T=T
    )
    return (normal_ ** 2).mean(dim=0)