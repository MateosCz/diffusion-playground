from abc import ABC, abstractmethod
from typing import Optional
import torch.nn as nn
import torch

class Schedule(ABC, nn.Module):
    @abstractmethod
    def beta(self, t: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def integral_beta(self, t: torch.Tensor):
        raise NotImplementedError

class LinearSchedule(Schedule):
    def __init__(self, beta_min: float, beta_max: float):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor):
        """
        Beta(t) = beta_min + (beta_max - beta_min) * t
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t
    
    def integral_beta(self, t: torch.Tensor):
        """
        Integral of beta(t) from 0 to t
        """
        return self.beta_min * t + (self.beta_max - self.beta_min) * t**2 / 2



class BaseSDE(ABC, nn.Module):
    """
    Base SDE class
    dx = f(x, t) dt + g(x,t) dW
    where f(x, t) is the drift and g(x, t) is the diffusion term
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def diffusion(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError
    
    def reverse_drift(
        self, 
        zt: torch.Tensor, 
        t: torch.Tensor, 
        score: torch.Tensor,
        eta: Optional[torch.Tensor | float] = None):
        raise NotImplementedError

class VPSDE(BaseSDE):
    """
    Variance Preserving SDE (VP-SDE)
    dx = f(x, t) dt + g(t) dW
    where f(x, t) = -0.5 * beta(t) * x
    and g(t) = sqrt(beta(t))
    """
    def __init__(self, schedule: Schedule):
        super().__init__()
        self.schedule = schedule

    def drift(self, x: torch.Tensor, t: torch.Tensor):
        beta = self.schedule.beta(t)
        return -0.5 * beta * x

    def diffusion(self, t: torch.Tensor):
        beta = self.schedule.beta(t)
        return torch.sqrt(beta)
            

    



class BaseIntegrator(ABC, nn.Module):
    def __init__(self, sde: BaseSDE):
        super().__init__()
        self.sde = sde

    @abstractmethod
    def integrate(self, x, t):
        raise NotImplementedError

class BaseSampler(ABC, nn.Module):
    def __init__(self, sde: BaseSDE, integrator: BaseIntegrator):
        super().__init__()
        self.sde = sde
        self.integrator = integrator

    @abstractmethod
    def sample(self, x, t):
        raise NotImplementedError

