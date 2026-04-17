"""
diffusion models

"""
from torch import nn
import torch
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence
from src.sde import VPSDE, BaseSDE, BaseSDEIntegrator, EulerIntegrator, LinearSchedule
from src.data import pos_to_angle


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
Constructor accepts 3 parameters: the backbone sde and the dimension of input data.
The sde in KLDM paper and TDM paper used for v is 
dx = -gamma*v_t*dt + sqrt(2 * gamma) dw
where they set gamma=1 same like 
the sde VPSDE(LinearSchedule(beta_min=2, beta_max=2)).
"""

class TDMDiffusion(BaseDiffusion):
    def __init__(self, dim: Sequence[int] | int, integrator_type: Literal["Exp, Euler"], sde: Optional[BaseSDE | None] = None):
        
        if sde is None:
            self.sde = VPSDE(LinearSchedule(beta_min=2,beta_max=2))
        self.sde = sde
        self.dim = dim
        self.integrator = self._get_integrator_by_name(integrator_type, sde)


    def _get_integrator_by_name(self, integrator_type, sde):
        if integrator_type is "Euler":
            integrator = EulerIntegrator(sde)
        return integrator
    
    """
    Sample forward(unconditioned) process for training.
    input is data f in Lie group SO(2)^n,total time, time's distribution keywords, 
    velocity's prior distribution keywords
    return the training pair (ft,vt, t) and the corresponding score (scoref, scorev)
    """

    def sample_forward(
        self,
        f0: torch.Tensor, # input Lie group data, with shape (batch_size, n_points, dim, 2 ,2) each point in SO(2)^dims
        total_time: float, 
        t_dist_kw: Literal["uniform"]="uniform", 
        v0_dist_kw: Literal["stdGauss", "zero"] = "zero" # usually initialized with v0 = 0
        ):
        batch_size = f0.shape[0]
        n_points = f0.shape[1]
        dim = f0.shape[3]

        # sample time uniformly
        if t_dist_kw is "uniform":
            ts = torch.rand(size=f0.shape[:2]) * total_time
        ts = torch.unsqueeze(ts,dim=-1)
        if v0_dist_kw is "zero":
            v0s = torch.zeros(size=f0.shape[:3])
        else:
            v0s = torch.randn(size=f0.shape[:3])
        # calculate mu and sigma for sampling vt
        mu_vt = self.sde.mean_t_coeff(ts)
        sigma_vt = self.sde.sigma_t(ts)

        # sampling vt
        epsv = torch.randn(size=v0s.shape)
        vts = epsv*sigma_vt + mu_vt

        # obtain score of v
        if v0_dist_kw is "zero":
            scorev = -vts/(sigma_vt**2)
        else:
            scorev = -epsv/sigma_vt

        #sample r given v
        rt = 

    """
    Sampling rt given vt,v0
    """
    def _sample_r_given_v(self, vt, v0,t):
        mu_rt = ((1-torch.exp(-t))/(1+torch.exp(t))) * (vt+v0)
        sigma_rt = 2*t + 8/(torch.exp(t)-1) -4
        epsrt = torch.randn(size=vt.shape)
        rt_raw = sigma_rt * epsrt + mu_rt
        wrapped_rt = pos_to_angle(rt_raw, torch.min(rt_raw), torch.max(rt_raw))





    def _sample_vt_given_v0(self,v0,t):
        mu_vt = self.sde.mean_t_coeff(t)
        sigma_vt = self.sde.sigma_t(t)

        # sampling vt
        epsv = torch.randn(size=v0.shape)
        vt = epsv*sigma_vt + mu_vt
        return vt

        
            


        






    def sample_forward_sim(self):

    def forward_kernel(self):

    def sample_backward(self):
    









