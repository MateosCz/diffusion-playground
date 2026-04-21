from abc import ABC, abstractmethod
from typing import Literal, Optional, Callable
from numpy import int16
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
        return reverse_drift

    

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
        sigma_t = torch.sqrt(1-torch.exp(-beta_integral))
        return sigma_t
        
        
            

    



class BaseSDEIntegrator(ABC, nn.Module):
    def __init__(self, sde: BaseSDE):
        super().__init__()
        self.sde = sde
    
    """
    integrate method in BaseSDEIntegrator class, must accept start data x0, 
    start time t0, end time T, number of time steps n_steps, score function score_fn
    Used to sample from back to begining, during inference step
    """
    @abstractmethod
    def integrate(
        self, 
        x0: torch.Tensor, 
        t0: torch.Tensor, 
        T: torch.Tensor, 
        n_steps: int, 
        direction: Literal["forward", "backward"],
        score_fn: Optional[Callable[[torch.Tensor,torch.Tensor],torch.Tensor]] = None,
        **kwargs):
        raise NotImplementedError

class EulerIntegrator(BaseSDEIntegrator):
    def __init__(self, sde: BaseSDE):
        super().__init__(sde)
        # self.sde = sde
        """
        Simplest integrator for SDEs, integrate along time t, compute marginal p_xt.
        Both accept forward sde integral and backward integral, depends on whether score_fn is given.
        """

    def integrate(
        self, 
        x0: torch.Tensor, 
        t0: torch.Tensor, 
        T: torch.Tensor, 
        n_steps: int, 
        direction: Literal["forward", "backward"],
        score_fn: Optional[Callable[[torch.Tensor,torch.Tensor],torch.Tensor]] = None, 
        **kwargs):
        if direction is "forward":
            integral_xt = self._integrate_forward(x0,t0,T,n_steps)
        else:
            integral_xt = self._integrate_backward(x0,t0,T,n_steps,score_fn)
        return integral_xt
        
    def integrate_forward_step(
        self,
        x_curr: torch.Tensor,
        t_curr: torch.Tensor,
        dt: torch.Tensor,
        **kwargs):
        dW = torch.sqrt(dt)*torch.randn_like(x_curr)
        x_next = x_curr + self.sde.drift(x_curr, t_curr) * dt + self.sde.diffusion(x_curr,t_curr) * dW
        return x_next

    def integrate_backward_step(
        self,
        x_curr: torch.Tensor,
        t_curr: torch.Tensor,
        score: torch.Tensor,
        dt: torch.Tensor,
        **kwargs):
        dW = torch.sqrt(dt)*torch.randn_like(x_curr)
        x_next = x_curr + self.sde.reverse_drift(x_curr, t_curr, score) * (-dt) + self.sde.diffusion(x_curr,t_curr) * dW
        return x_next

        
    def _integrate_forward(        
        self, 
        x0: torch.Tensor, 
        t0: torch.Tensor, 
        T: torch.Tensor, 
        n_steps: int, 
        **kwargs):
        dt = (T - t0)/ n_steps
        xt = x0.clone()
        t = t0.clone()
        for _ in range(n_steps):
            dW = torch.sqrt(dt)*torch.randn_like(x0)
            xt = xt + self.sde.drift(xt, t) * dt + self.sde.diffusion(xt,t) * dW
            t = t + dt
        
        return xt





    def _integrate_backward(
        self, 
        x0: torch.Tensor, 
        t0: torch.Tensor, 
        T: torch.Tensor, 
        n_steps: int, 
        score_fn: Callable[[torch.Tensor,torch.Tensor],torch.Tensor], 
        **kwargs):
        dt = (t0 - T)/ n_steps
        zt = x0.clone()
        t = t0.clone()
        for _ in range(n_steps):
            dW = torch.sqrt(dt)*torch.randn_like(x0)
            score = score_fn(zt,t)
            zt = zt + self.sde.reverse_drift(zt, t, score) * (-dt) + self.sde.diffusion(zt,t)*dW
            t = t-dt
        return zt