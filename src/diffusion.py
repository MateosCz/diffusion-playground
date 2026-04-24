"""
diffusion models

"""
from torch import nn
import torch
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence, Callable, Tuple
from src.sde import VPSDE, BaseSDE, BaseSDEIntegrator, EulerIntegrator, LinearSchedule
from src.data import pos_to_angle, wrap_angle
from src.distribution import WrappedNormalDistribution



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
Input data should already be wrapped to LieTorus angle data (data in [-pi, pi)^n).
Constructor accepts 3 parameters: the backbone sde and the dimension of input data.
The sde in KLDM paper and TDM paper used for v is 
dx = -gamma*v_t*dt + sqrt(2 * gamma) dw
where they set gamma=1 same like 
the sde VPSDE(LinearSchedule(beta_min=2, beta_max=2)).
"""

class TDMDiffusion(BaseDiffusion):
    def __init__(
        self, 
        dim: Sequence[int] | int, 
        integrator_type: Literal["Exp", "Euler"], 
        sde: Optional[BaseSDE | None] = None, 
        trunc_n: int = 10,
        f_scale: float = 2 * torch.pi
        ):
        super().__init__()
        if sde is None:
            self.sde = VPSDE(LinearSchedule(beta_min=2,beta_max=2))
        else:
            self.sde = sde
        self.dim = dim
        self.integrator = self._get_integrator_by_name(integrator_type, self.sde)
        self.trunc_n = trunc_n
        self.f_scale = f_scale


    def _get_integrator_by_name(self, integrator_type, sde):
        if integrator_type == "Euler":
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
        f0: torch.Tensor, # input Lie group data in angle form, with shape (batch_size, n_points, dim) each point in [-pi, pi)^dim
        total_time: float, 
        t_dist_kw: Literal["uniform", "linear", "constant"]="uniform", 
        v0_dist_kw: Literal["stdGauss", "zero"] = "zero", # usually initialized with v0 = 0
        n_steps: int = 100, # number of time steps if t_dist_kw is "linear"
        constant_t: float = 1.0, # constant time if t_dist_kw is "constant"
        return_time: bool = False # whether to return the time tensor for training
        ):
        device = f0.device
        dtype = f0.dtype
        batch_size = f0.shape[0]
        n_points = f0.shape[1]
        dim = f0.shape[2]

        # sample time uniformly
        if t_dist_kw == "uniform":
            ts = torch.rand(size=(batch_size,), device=device, dtype=dtype) * total_time # shape (batch_size)
            ts = ts.view(batch_size,1,1).expand(batch_size, n_points, dim) # shape (batch_size, n_points, dim)
        elif t_dist_kw == "linear":
            ts = torch.linspace(0, total_time, n_steps, device=device, dtype=dtype) # shape (n_steps)
            ts = torch.unsqueeze(ts,dim=0).repeat(f0.shape[0],1) # shape (batch_size, n_steps)
        elif t_dist_kw == "constant":
            ts = constant_t * torch.ones(size=(batch_size, n_points, dim), device=device, dtype=dtype)
        if v0_dist_kw == "zero":
            v0s = torch.zeros(size=f0.shape, device=device, dtype=dtype) # shape (batch_size, n_points, dim)
        else:
            v0s = torch.randn(size=f0.shape, device=device, dtype=dtype) # shape (batch_size, n_points, dim)
        # calculate mu and sigma for sampling vt
        mu_vt = self.sde.mean_t_coeff(ts) * v0s
        sigma_vt = self.sde.sigma_t(ts)

        # sampling vt
        epsv = torch.randn(size=v0s.shape, device=device, dtype=dtype)
        vts = epsv*sigma_vt + mu_vt

        # obtain score of v
        if v0_dist_kw == "zero":
            scorev = -vts/(sigma_vt**2)
        else:
            scorev = -epsv/sigma_vt

        #sample wrapped scalar rt given vt,v0,t
        wrapped_rts = self._sample_r_given_v(vts, v0s, ts)

        # fractional coordinate ft = f0expm(rt), 
        # Eq(15) shows f0expm(rt) = wrap(f0+rt)
        fts = wrap_angle(f0 + wrapped_rts)
        scorec = self._score_c(vts, v0s, ts, wrapped_rts)

        score = scorec + scorev
        latents = (vts, fts)
        # print(f"min sigma_rt={self._sigma_rt(ts).min()}, max sigma_rt={self._sigma_rt(ts).max()}")
        if return_time:
            return latents, score, ts[:,0,0] # (B,)
        # latents = (vts/ self.f_scale, fts/ self.f_scale) # (normalized vt, normalized ft)
        return latents, score



    """
    Sampling rt given vt,v0
    sample from wrapped normal distribution WN(r_t| rt_mu, rt_sigma)
    In paper rt is skew symmetric matrix, Rt = [[0, r_t], [-r_t, 0]]
    but we only sample the anti-diagonal scalar r_t

    """
    def _sample_r_given_v(self, vt, v0,t):
        mu_rt = self._mu_rt(vt, v0, t)
        sigma_rt = self._sigma_rt(t)
        epsrt = torch.randn(size=vt.shape, device=vt.device, dtype=vt.dtype)
        rt_raw = sigma_rt * epsrt + mu_rt
        wrapped_rt = wrap_angle(rt_raw) 
        return wrapped_rt

    def _mu_rt(self, vt, v0, t):
        return ((1-torch.exp(-t))/(1+torch.exp(-t))) * (vt+v0)

    def _sigma_rt(self, t, eps = 1e-6):
        return torch.sqrt(2*t + 8/(torch.exp(t)+1) -4 + eps)
        
    def _sample_vt_given_v0(self,v0,t):
        mu_coeff_vt = self.sde.mean_t_coeff(t)
        sigma_vt = self.sde.sigma_t(t)

        # sampling vt
        epsv = torch.randn(size=v0.shape, device=v0.device, dtype=v0.dtype)
        vt = epsv*sigma_vt + mu_coeff_vt * v0 #mu_coeff_vt is the coefficient of mu
        return vt

    def _score_c(self, vt,v0,t, rt):
        mu_rt = self._mu_rt(vt, v0, t)
        sigma_rt = self._sigma_rt(t)
        mu_rt = wrap_angle(mu_rt)
        WN_distribution = WrappedNormalDistribution(mu_rt, sigma_rt, self.trunc_n)
        scorec = (1-torch.exp(-t))/(1+torch.exp(-t)) * WN_distribution.score(rt)
        return scorec

    # DSM loss function for training the score network
    def loss_diffusion(self, pred, target, t):
        assert pred.shape == target.shape
        #without reweighting
        loss = torch.nn.functional.mse_loss(pred, target)
        return loss

    # DSM loss function for training the score network with reweighting
    def loss_diffusion_reweighting(self, pred, target, t):
        assert pred.shape == target.shape
        #with reweighting
        reweighting_term = self.sde.sigma_t(t)
        #map the reweighting_term to the same shape as pred and target
        reweighting_term = reweighting_term[:, None, None].expand(-1, pred.shape[1], pred.shape[2])
        loss = torch.nn.functional.mse_loss(pred * reweighting_term, target * reweighting_term)
        return loss

    


    def sample_backward(
        self,
        fT_prior_kw: Literal["stdGauss", "uniform"],
        vT_prior_kw: Literal["stdGauss", "uniform"],
        data_shape: Tuple[int, int, int],
        total_time: float,
        tdm_score_fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor], 
        # score function of TDM model, input f_t, v_t, t, return the score
        n_steps: int = 100,
        sample_trajectory: bool = False,
        **kwargs

    ):
        assert fT_prior_kw in ["stdGauss", "uniform"]
        assert vT_prior_kw in ["stdGauss", "uniform"]
        device = next(self.parameters()).device # obtain the device of the model from parameters
        dtype = next(self.parameters()).dtype # obtain the dtype of the model from parameters
        batch_size = data_shape[0]
        n_points = data_shape[1]
        dim = data_shape[2]

        dt = total_time / n_steps # calculate the dt
        # sample fT and vT from prior distribution
        if fT_prior_kw == "stdGauss":
            fT = torch.randn(size=(batch_size, n_points, dim), device=device, dtype=dtype)
        else:
            fT = torch.rand(size=(batch_size, n_points, dim), device=device, dtype=dtype)
            fT = pos_to_angle(fT)
            fT = wrap_angle(fT)
        if vT_prior_kw == "stdGauss":
            vT = torch.randn(size=(batch_size, n_points, dim), device=device, dtype=dtype)
        else:
            vT = torch.rand(size=(batch_size, n_points, dim), device=device, dtype=dtype)
        ft_reverse = fT
        vt_reverse = vT
        ft_reverse_trajectory = [ft_reverse]
        vt_reverse_trajectory = [vt_reverse]
        t_list = [total_time]

        # reverse time step by step
        for i in range(n_steps):
            t_reverse = total_time - i * dt
            score_total_reverse = tdm_score_fn(ft_reverse, vt_reverse, t_reverse)
            scorev_reverse = self._get_scorev_from_scorec(score_total_reverse, vt_reverse, t_reverse)
            eps_v = torch.randn(size=vt_reverse.shape, device=device, dtype=dtype)

            # using exponential integration to solve the sde backward step
            vt_reverse = torch.exp(dt) * vt_reverse + 2 * (torch.exp(2*dt)-1)* scorev_reverse + torch.sqrt(torch.exp(2 * dt) - 1) * eps_v
            ft_reverse = ft_reverse + dt * vt_reverse
            ft_reverse = wrap_angle(ft_reverse)
            ft_reverse_trajectory.append(ft_reverse)
            vt_reverse_trajectory.append(vt_reverse)
            t_list.append(t_reverse)
        if sample_trajectory:
            return ft_reverse_trajectory, vt_reverse_trajectory, t_list
        else:
            return ft_reverse, vt_reverse
            



    #Eq(19) in KLDM paper, v0 is 0
    def _get_scorev_from_scorec(self, scorec, vt, t): 
        scorev = self._prefector(t) * scorec - vt/(self.sde.sigma_t(t)**2) # only for v0 = 0
        return scorev

    def _get_prefector(self, t):
        return (1-torch.exp(-t))/(1+torch.exp(-t))