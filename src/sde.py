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
    

    """
    transition kernel parameters
    """
    @abstractmethod
    def mean_t_coeff(self, t:torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def sigma_t(self, t:torch.Tensor):
        raise NotImplementedError

    
    """
    function for computing the time-reversed sde's drift, given time reversed zt, t, score, eta 
    """
    def reverse_drift(
        self, 
        zt: torch.Tensor, 
        t: torch.Tensor, 
        score: torch.Tensor,
        eta: Optional[torch.Tensor | float] = None):
        f = self.drift(zt,t)
        g = self.diffusion(zt,t)
        """
        when eta is 0, then the reverse diffusion sample the probability flow ODE:
        dx = (f(x,t) - 1/2 g(x,t) * g(x,t)^T * score) dt
        rather than from SDE:
        dx = (f(x,t) - g(x,t) * g(x,t)^T * score) dt + g(x,t) dW
        
        """
        if eta is None: 
            eta = g
        reverse_drift = f - 0.5 * (g**2 + eta **2) * score

    

class VPSDE(BaseSDE):
    """
    Variance Preserving SDE (VP-SDE)
    dx = f(x, t) dt + g(t) dW
    where f(x, t) = -0.5 * beta(t) * x
    and g(t) = sqrt(beta(t))
    transition kernel pt|0 = N(x_t| alpha_t * x_0, sigma_t**2 * I)
    alpha_t is mean_t_coeff
    sigma_t is sigma_t
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

    
    """
    transition kernel parameters, mean_t_coeff(alpha_t) and sigma_t(sigma_t)
    """
    
    def mean_t_coeff(self, t: torch.Tensor):
        beta_integral = self.schedule.integral_beta(t)
        mean_t_coeff = torch.exp(-0.5 * beta_integral)
        return mean_t_coeff

    def sigma_t(self, t:torch.Tensor):
        beta_integral = self.schedule.integral_beta(t)
        sigma_t = torch.sqrt(1-torch.exp(beta_integral))
        
        
            

    



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

