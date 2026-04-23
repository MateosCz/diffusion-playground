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
            C_component = torch.exp((-(x - self.mu + k * T)**2)/(2*self.sigma**2))
            C2_component = (x - self.mu + k *T) / self.sigma**2
            C_2 += C_component * C2_component
            C += C_component
        
        return C_2 / C
            








    # def sample(self):


