"""
diffusion models

"""
from torch import nn
import torch
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence, Callable, Tuple
from src.sde import VPSDE, BaseSDE, BaseSDEIntegrator, EulerIntegrator, LinearSchedule
from src.data import pos_to_angle, wrap_angle, wrap_pos
from src.distribution import WrappedNormalDistribution, sigma_norm
import math
from torch_geometric.data import Data
from torch_geometric.utils import scatter



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
        trunc_n: int = 13,
        f_scale: float = 2 * torch.pi,
        total_time: float = 2.0,
        simplified_param: bool = True,
        n_sigma_rs:int = 2000
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
        self.simplified_param = simplified_param
        self.n_sigma_rs = n_sigma_rs
        self.total_time = total_time
        if simplified_param:
            sigma_rs = self._sigma_rt(torch.linspace(0, total_time, self.n_sigma_rs))
            sigma_norms_r = sigma_norm(sigma_rs, T=2 * torch.pi, N=self.trunc_n, sn=self.n_sigma_rs)
            self.register_buffer("sigma_norms_r", sigma_norms_r)


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
        f0: torch.Tensor, # input Lie group data in angle form, with shape (batch_size, dim) each point in [-pi, pi)^dim
        total_time: float, 
        t_dist_kw: Literal["uniform", "quadratic", "linear", "constant"]="uniform", 
        v0_dist_kw: Literal["stdGauss", "zero"] = "zero", # usually initialized with v0 = 0
        n_steps: int = 100, # number of time steps if t_dist_kw is "linear"
        constant_t: float = 1.0, # constant time if t_dist_kw is "constant"
        return_time: bool = False, # whether to return the time tensor for training
        t_min: float = 5e-3,
        ):
        device = f0.device
        dtype = f0.dtype
        batch_size = f0.shape[0]
        dim = f0.shape[1]

        # sample time uniformly
        if t_dist_kw == "uniform":
            ts = torch.rand(size=(batch_size,1), device=device, dtype=dtype) # shape (batch_size, 1)
            # ts = ts.view(batch_size,1).expand(batch_size, dim) # shape (batch_size, dim)
            ts = ts * (total_time - t_min) + t_min
        elif t_dist_kw == "quadratic":
            ts = torch.rand(size=(batch_size,1), device=device, dtype=dtype) ** 2
            ts = ts * (total_time - t_min) + t_min
        elif t_dist_kw == "linear":
            ts = torch.linspace(0, total_time, n_steps, device=device, dtype=dtype) # shape (n_steps)
            ts = torch.unsqueeze(ts,dim=0).repeat(batch_size,1) # shape (batch_size, n_steps)
        elif t_dist_kw == "constant":
            ts = constant_t * torch.ones(size=(batch_size,1), device=device, dtype=dtype)
        if v0_dist_kw == "zero":
            v0s = torch.zeros(size=f0.shape, device=device, dtype=dtype) # shape (batch_size, dim)
        else:
            v0s = torch.randn(size=f0.shape, device=device, dtype=dtype) # shape (batch_size, dim)
        # calculate mu and sigma for sampling vt
        mu_vt = self.sde.mean_t_coeff(ts) * v0s
        sigma_vt = self.sde.sigma_t(ts)

        # sampling vt
        epsv = torch.randn(size=v0s.shape, device=device, dtype=dtype)
        # epsv = self.center_data(epsv)

        vts = epsv*sigma_vt + mu_vt

        # obtain score of v
        if v0_dist_kw == "zero":
            scorev = -vts/(sigma_vt**2)
            # scorev = -epsv/sigma_vt
        else:
            scorev = -epsv/sigma_vt

        #sample wrapped scalar rt given vt,v0,t
        wrapped_rts = self._sample_r_given_v(vts, v0s, ts)

        # fractional coordinate ft = f0expm(rt), 
        # Eq(15) shows f0expm(rt) = wrap(f0+rt)
        fts = wrap_angle(f0 + wrapped_rts)
        # fts = self.center_data(fts)


        # return the whole score(scorec + scorev), scorec is from Eq(26) in KLDM paper
        

        # only return the score of wrapped normal distribution(\nabla log p(r_t))
        if self.simplified_param:
            scorec = self._score_c(vts, v0s, ts, wrapped_rts, with_prefector=False)
            sigma_norm_r_t = torch.sqrt(self._sigma_norm_t(ts))

            score = scorec / sigma_norm_r_t
            # score = scorec
        else:
            scorec = self._score_c(vts, v0s, ts, wrapped_rts, with_prefector=True)
            score = scorec + scorev
        latents = (vts, fts)
        # print(f"min sigma_rt={self._sigma_rt(ts).min()}, max sigma_rt={self._sigma_rt(ts).max()}")
        if return_time:
            return latents, score, ts # (B, 1)
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
        # epsrt = self.center_data(epsrt)
        rt_raw = sigma_rt * epsrt + mu_rt
        wrapped_rt = wrap_angle(rt_raw) 
        return wrapped_rt

    def _mu_rt(self, vt, v0, t):
        return ((1-torch.exp(-t))/(1+torch.exp(-t))) * (vt+v0)

    def _sigma_rt(self, t, eps = 1e-6):
        return torch.sqrt(2*t + 8/(torch.exp(t)+1) -4 + eps)
        
    def _sample_vt_given_v0(self,v0,t, dt):
        mu_coeff_vt = self.sde.mean_t_coeff(t)
        sigma_vt = self.sde.sigma_t(t)

        # sampling vt
        epsv = torch.randn(size=v0.shape, device=v0.device, dtype=v0.dtype)
        vt = epsv*sigma_vt + mu_coeff_vt * v0 #mu_coeff_vt is the coefficient of mu
        return vt

    def _score_c(self, vt,v0,t, rt, with_prefector=True):
        mu_rt = self._mu_rt(vt, v0, t)
        sigma_rt = self._sigma_rt(t)
        # mu_rt = wrap_angle(mu_rt)
        WN_distribution = WrappedNormalDistribution(mu_rt, sigma_rt, self.trunc_n)
        if with_prefector:
            scorec = (1-torch.exp(-t))/(1+torch.exp(-t)) * WN_distribution.score(rt)
        else:
            scorec = WN_distribution.score(rt)
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
        exponential_integration: bool = True,
        probability_flow: bool = False,
        predictor_corrector: bool = False,
        predictor_corrector_n_steps: int = 1,
        only_correct_vt: bool = False,
        tau: float = 1e-3,
        vT_prior_scale: float = 1.0,
        annealing_gamma: float = 1.0,
        debug: bool = False,    
        **kwargs

    ):
        assert fT_prior_kw in ["stdGauss", "uniform"]
        assert vT_prior_kw in ["stdGauss", "uniform"]

        device, dtype = torch.device("cpu"), torch.float32

        batch_size = data_shape[0]
        dim = data_shape[1]

        dt = total_time / (n_steps-1) # calculate the dt
        # sample fT and vT from prior distribution
        if fT_prior_kw == "stdGauss":
            fT = torch.randn(size=(batch_size, dim), device=device, dtype=dtype)
        else:
            fT = torch.rand(size=(batch_size, dim), device=device, dtype=dtype)
            fT = pos_to_angle(fT)
            fT = wrap_angle(fT)
        if vT_prior_kw == "stdGauss":
            vT = torch.randn(size=(batch_size, dim), device=device, dtype=dtype) * vT_prior_scale
        else:
            vT = torch.rand(size=(batch_size, dim), device=device, dtype=dtype)
        ft_reverse = fT
        vt_reverse = vT
        ft_reverse_trajectory = []
        ft_reverse_trajectory.append(ft_reverse)
        vt_reverse_trajectory = []
        vt_reverse_trajectory.append(vt_reverse)
        t_list = []
        t_reverse = torch.ones((batch_size,1), device=device, dtype=dtype) * total_time # shape (batch_size, 1)
        t_list.append(float(t_reverse[0, 0].detach().cpu()))
        dt = torch.tensor(dt, device=device, dtype=dtype)
        sigma_list = []
        sigma_list.append(float(self.sde.sigma_t(t_reverse)[0, 0].detach().cpu()))
        t_reverse_list = torch.linspace(total_time, 0, n_steps, device=device, dtype=dtype)
        t_reverse_list = torch.unsqueeze(t_reverse_list, dim=0).repeat(batch_size, 1) # shape (batch_size, n_steps)
        t_reverse_list = t_reverse_list.view(batch_size, n_steps, 1) # shape (batch_size, n_steps, 1)
        # reverse time step by step
        for i in range(n_steps-1):
            t_reverse = t_reverse_list[:,i] # shape (batch_size, 1)
            t_reverse_next = t_reverse_list[:,i+1] # shape (batch_size, 1)
            score_learned = tdm_score_fn(ft_reverse, vt_reverse, t_reverse)
            eps_v = torch.randn(size=vt_reverse.shape, device=device, dtype=dtype)
            if self.simplified_param:
                prefactor = self._get_prefector(t_reverse)
                sigma_norm_r_t = torch.sqrt(self._sigma_norm_t(t_reverse))
                sigma_v = self.sde.sigma_t(t_reverse)
                # scorev_reverse = score_learned * prefactor * sigma_norm_r_t - vt_reverse / (sigma_v**2)
                scorev_reverse = score_learned * prefactor * sigma_norm_r_t - vt_reverse / (sigma_v**2)


            else:
                scorev_reverse = score_learned       

            # if i == n_steps-2:
            #     # use Tweedie's formula at the last time step
            #     ft_reverse, vt_reverse = self._tweedie_step(ft_reverse, vt_reverse, t_reverse, dt, tdm_score_fn)
            # else:
                 
                # predictor-corrector sampling strategy
            if predictor_corrector:
                # only using probability flow ODE as predictor step
                if exponential_integration:
                    vt_reverse = (torch.exp(dt) * vt_reverse +
                                    torch.expm1(dt) * scorev_reverse)
                    # r = self.sde.mean_t_coeff(t_reverse_next) / self.sde.mean_t_coeff(t_reverse)
                    # prefactor_em = (r * self.sde.sigma_t(t_reverse) - self.sde.sigma_t(t_reverse_next)) * self.sde.sigma_t(t_reverse)
                    # vt_reverse = vt_reverse * r + prefactor_em * scorev_reverse 
                else:
                    vt_reverse = (vt_reverse - 
                    (-vt_reverse - scorev_reverse) * dt)
            
            
            # probability flow sampling strategy
            elif probability_flow:
                if exponential_integration:
                    vt_reverse = (torch.exp(dt) * vt_reverse +
                                 (torch.expm1(dt)) * scorev_reverse)
                else:
                    vt_reverse = vt_reverse - (-vt_reverse - scorev_reverse) * dt
            
            # conventional SDE sampling strategy
            else:
                if exponential_integration:
                    vt_reverse = (torch.exp(dt) * vt_reverse +
                                2 * (torch.expm1(dt)) * scorev_reverse + 
                                torch.sqrt(torch.expm1(2 * dt)) * eps_v)
                else:
                    vt_reverse = vt_reverse - (-vt_reverse - 2 * scorev_reverse) * dt + torch.sqrt(2 * dt) * eps_v
            # update ft_reverse to next time step
            ft_reverse = ft_reverse - dt * vt_reverse
            ft_reverse = wrap_angle(ft_reverse)
            
            # debugging information
            
            
            # corrector step if not one time step before last time step
            if predictor_corrector:
                if i <n_steps-2:
                    corr_step_size = tau * t_reverse_next **2 / self.total_time **2
                    if not only_correct_vt:
                        for _ in range(predictor_corrector_n_steps):
                            ft_reverse, vt_reverse = self._langevin_corrector_step(ft_reverse, vt_reverse, t_reverse_next, dt, tdm_score_fn, tau=corr_step_size)
                    elif only_correct_vt:
                        for _ in range(predictor_corrector_n_steps):
                            vt_reverse = self._langevin_corrector_step(ft_reverse, vt_reverse, t_reverse_next, dt, tdm_score_fn, only_correct_vt=True, tau=tau)
            sigma_list.append(float(self.sde.sigma_t(t_reverse_next)[0, 0].detach().cpu()))
            # sample trajectory if needed
            if sample_trajectory:
                ft_reverse_trajectory.append(ft_reverse)
                vt_reverse_trajectory.append(vt_reverse)

        # debugging information
        t_reverse_list = t_reverse_list[0].reshape(-1).detach().cpu().numpy()
        sigma_list = torch.tensor(sigma_list).reshape(-1).numpy()
        if debug:
            return ft_reverse_trajectory, vt_reverse_trajectory, t_reverse_list, sigma_list
        else:
            if sample_trajectory:
                return ft_reverse_trajectory, vt_reverse_trajectory, t_reverse_list

            else:
                return ft_reverse, vt_reverse
        
    
    
    def _langevin_corrector_step(
        self,
        ft_reverse,
        vt_reverse,
        t_reverse,
        dt: torch.Tensor,
        score_fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor],
        tau: float = 0.25,
        only_correct_vt: bool = False
        ):
        """
        Correction step using Langevin dynamics
        only for simplified parameter case
        """
        with torch.no_grad():
            score_learned = score_fn(ft_reverse, vt_reverse, t_reverse)
        device, dtype = vt_reverse.device, vt_reverse.dtype
        tau = torch.tensor(tau, device=device, dtype=dtype)
        eps_v = torch.randn(size=vt_reverse.shape, device=device, dtype=dtype)
        prefactor = self._get_prefector(t_reverse)
        sigma_norm_r_t = torch.sqrt(self._sigma_norm_t(t_reverse))
        sigma_v = self.sde.sigma_t(t_reverse)
        mean_t_coeff = self.sde.mean_t_coeff(t_reverse)
        scorev_reverse = score_learned * prefactor * sigma_norm_r_t - vt_reverse / (sigma_v**2)
        # alpha_t = mean_t_coeff**2
        # r = tau
        # delta = 2 * alpha_t * ((r * eps_v.square()/scorev_reverse.square()) ** 2)
        # denom = scorev_reverse.square().mean(dim=-1, keepdim=True)
        # sigma_factor = sigma_v**2 / self.sde.sigma_t(torch.tensor(self.total_time, device=device, dtype=dtype))**2
        # delta = tau / denom * sigma_factor
        delta = tau
        vt_reverse = vt_reverse + delta * scorev_reverse + torch.sqrt(2 * delta) * eps_v
        
        if not only_correct_vt:
            ft_reverse = ft_reverse - dt * vt_reverse
            ft_reverse = wrap_angle(ft_reverse)
            return ft_reverse, vt_reverse
        else:
            return vt_reverse

    def _tweedie_step(
        self,
        ft_reverse,
        vt_reverse,
        t_reverse,
        dt: torch.Tensor,
        score_fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor],
        only_correct_vt: bool = False
    ):
        """
        Correction step using Tweedie's formula, for one time step before last time step
        only for simplified parameter case
        """

        # reconstruct the score
        with torch.no_grad():
            score_learned = score_fn(ft_reverse, vt_reverse, t_reverse)
        device, dtype = vt_reverse.device, vt_reverse.dtype
        prefactor = self._get_prefector(t_reverse)
        sigma_norm_r_t = torch.sqrt(self._sigma_norm_t(t_reverse))
        sigma_v = self.sde.sigma_t(t_reverse)
        scorev_reverse = score_learned * prefactor * sigma_norm_r_t - vt_reverse / (sigma_v**2)

        # update vt using Tweedie's formula
        v0 = vt_reverse + dt * scorev_reverse
        if not only_correct_vt:
            f0 = ft_reverse - dt * v0
        else:
            f0 = ft_reverse
        f0 = wrap_angle(f0)
        # f0 = ft_reverse
        return f0, v0

    #Eq(19) in KLDM paper, v0 is 0
    def _get_scorev_from_scoreall(self, scoreall, vt, t): 
        scorev = self._get_prefector(t) * scoreall - vt/(self.sde.sigma_t(t)**2) # only for v0 = 0
        return scorev

    def _get_prefector(self, t):
        return (1-torch.exp(-t))/(1+torch.exp(-t))

    """
    make the sum of data in each dimension to be 0
    data: (batch_size, 1, dim)
    """
    def center_data(self, data):
        return data - torch.mean(data, dim=0, keepdim=True)

    def _sigma_norm_t(self, t: torch.Tensor):
        idx = torch.round(t / self.total_time * (len(self.sigma_norms_r))).long() - 1
        return self.sigma_norms_r[idx]
    

    # ------------------------------------------------------------------
    # Graph-aware forward / backward sampling
    # ------------------------------------------------------------------

    def sample_forward_graph(
        self,
        graph: Data,
        total_time: float,
        t_dist_kw: Literal["uniform", "quadratic", "constant"] = "uniform",
        v0_dist_kw: Literal["stdGauss", "zero"] = "zero",
        constant_t: float = 1.0,
        return_time: bool = False,
        t_min: float = 5e-3,
    ):
        """
        Graph-structured forward (noising) process for training.

        Accepts a PyG Batch produced by PyGGraphWrapper (or a DataLoader over
        it).  One diffusion time is sampled per graph and shared across all
        nodes of that graph; the actual noising arithmetic mirrors
        ``sample_forward`` exactly.

        Parameters
        ----------
        graph : torch_geometric.data.Data / Batch
            Must have ``graph.x`` (N_total, 2) node angles in [-pi, pi),
            ``graph.batch`` (N_total,) graph-membership indices, and valid
            ``graph.edge_index`` / ``graph.edge_attr``.
        total_time : float
        t_dist_kw : "uniform" | "quadratic" | "constant"
        v0_dist_kw : "stdGauss" | "zero"
        constant_t : float
            Used only when t_dist_kw == "constant".
        return_time : bool
            When True also return ``ts`` of shape (num_graphs, 1).
        t_min : float

        Returns
        -------
        (vts, noised_graph), score          when return_time=False
        (vts, noised_graph), score, ts      when return_time=True

        vts         : (N_total, 2)  velocity at each node
        noised_graph: graph with .x / .pos / .edge_attr updated to f_t
        score       : (N_total, 2)  target score at each node
        ts          : (num_graphs, 1)
        """
        device = graph.x.device
        dtype = graph.x.dtype
        batch_vec = graph.batch          # (N_total,)
        num_graphs = int(batch_vec.max().item()) + 1
        f0 = graph.x                     # (N_total, 2)

        # --- one time per graph, then expand to per-node ---
        if t_dist_kw == "uniform":
            ts = torch.rand(size=(num_graphs, 1), device=device, dtype=dtype)
            ts = ts * (total_time - t_min) + t_min
        elif t_dist_kw == "quadratic":
            ts = torch.rand(size=(num_graphs, 1), device=device, dtype=dtype) ** 2
            ts = ts * (total_time - t_min) + t_min
        elif t_dist_kw == "constant":
            ts = constant_t * torch.ones(size=(num_graphs, 1), device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown t_dist_kw: {t_dist_kw}")

        ts_node = ts[batch_vec]          # (N_total, 1) time per node, each sample share a same time

        # --- initial velocity ---
        if v0_dist_kw == "zero":
            v0s = torch.zeros_like(f0)
        else:
            v0s = torch.randn_like(f0)

        # --- velocity forward process ---
        mu_vt = self.sde.mean_t_coeff(ts_node) * v0s
        sigma_vt = self.sde.sigma_t(ts_node)
        epsv = torch.randn_like(v0s)
        vts = epsv * sigma_vt + mu_vt

        if v0_dist_kw == "zero":
            scorev = -vts / (sigma_vt ** 2)
        else:
            scorev = -epsv / sigma_vt

        # --- position forward process ---
        wrapped_rts = self._sample_r_given_v(vts, v0s, ts_node)
        fts = wrap_angle(f0 + wrapped_rts)

        # --- target score ---
        if self.simplified_param:
            scorec = self._score_c(vts, v0s, ts_node, wrapped_rts, with_prefector=False)
            sigma_norm_r_t = torch.sqrt(self._sigma_norm_t(ts_node))
            score = scorec / sigma_norm_r_t
        else:
            scorec = self._score_c(vts, v0s, ts_node, wrapped_rts, with_prefector=True)
            score = scorec + scorev

        # --- refresh graph with noised positions ---
        # noised_graph = self._update_graph_batch(graph, fts)
        noised_graph = graph.clone()
        noised_graph = self._update_graph_batch(noised_graph, fts)



        if return_time:
            return (vts, noised_graph), score, ts
        return (vts, noised_graph), score

    @torch.inference_mode()
    def sample_backward_graph(
        self,
        graph_T: Data,
        vT: torch.Tensor,
        total_time: float,
        tdm_score_fn: Callable[[Data, torch.Tensor, torch.Tensor], torch.Tensor],
        n_steps: int = 100,
        sample_trajectory: bool = False,
        exponential_integration: bool = True,
        probability_flow: bool = False,
        predictor_corrector: bool = False,
        predictor_corrector_n_steps: int = 1,
        only_correct_vt: bool = False,
        tau: float = 1e-3,
        annealing_gamma: float = 1.0,
        debug: bool = False,
    ):
        """
        Graph-structured reverse (denoising) process for sampling.

        Mirrors ``sample_backward`` exactly but operates on node-level
        tensors so that the score network can exploit the graph structure
        at every step.

        Parameters
        ----------
        graph_T : torch_geometric.data.Data / Batch
            Graph batch whose ``.x`` / ``.pos`` are already set to the
            prior samples ``f_T`` (angles in [-pi, pi)).  Must also carry
            ``graph_T.batch`` and valid ``edge_index`` / ``edge_attr``.
        vT : (N_total, 2)
            Prior velocity at each node.
        total_time : float
        tdm_score_fn : callable  (graph, vt, t) -> (N_total, 2)
            Score network wrapper.  ``graph`` is the *updated* PyG Batch
            at the current reverse time step; ``vt`` is (N_total, 2); ``t``
            is (num_graphs, 1).
        n_steps : int
        sample_trajectory : bool
        exponential_integration : bool
        probability_flow : bool
        predictor_corrector : bool
        predictor_corrector_n_steps : int
        only_correct_vt : bool
        tau : float
        annealing_gamma : float
        debug : bool

        Returns
        -------
        (ft_reverse, vt_reverse)                           default
        (ft_traj, vt_traj, t_arr)                          sample_trajectory=True
        (ft_traj, vt_traj, t_arr, sigma_arr)               debug=True
        """
        device = graph_T.x.device
        dtype = graph_T.x.dtype
        batch_vec = graph_T.batch                           # (N_total,)
        num_graphs = int(batch_vec.max().item()) + 1

        dt_scalar = total_time / (n_steps - 1)
        dt = torch.tensor(dt_scalar, device=device, dtype=dtype)

        ft_reverse = graph_T.x.clone()                     # (N_total, 2)
        vt_reverse = vT.clone()                            # (N_total, 2)
        graph = graph_T                                     # mutable reference

        ft_traj, vt_traj, t_list, sigma_list = [], [], [], []
        if sample_trajectory:
            ft_traj.append(ft_reverse.cpu())
            vt_traj.append(vt_reverse.cpu())

        t_reverse_list = torch.linspace(total_time, 0, n_steps, device=device, dtype=dtype)
        # shape: (num_graphs, n_steps, 1) — same time for all graphs at each step
        t_reverse_list = t_reverse_list.unsqueeze(0).unsqueeze(-1).expand(num_graphs, -1, -1)
        t_list.append(float(t_reverse_list[0, 0, 0].item()))
        sigma_list.append(float(self.sde.sigma_t(t_reverse_list[:, 0])[0, 0].item()))

        for i in range(n_steps - 1):
            # per-graph time: (num_graphs, 1)
            t_g = t_reverse_list[:, i, :]       # (num_graphs, 1)
            t_g_next = t_reverse_list[:, i + 1, :]

            # per-node time: (N_total, 1)
            t_n = t_g[batch_vec]

            score_learned = tdm_score_fn(graph, vt_reverse, t_g)   # (N_total, 2)
            eps_v = torch.randn_like(vt_reverse)

            if self.simplified_param:
                prefactor = self._get_prefector(t_n)                # (N_total, 1)
                sigma_norm_r_t = torch.sqrt(self._sigma_norm_t(t_n))
                sigma_v = self.sde.sigma_t(t_n)
                scorev_reverse = (score_learned * prefactor * sigma_norm_r_t
                                  - vt_reverse / (sigma_v ** 2))
            else:
                scorev_reverse = score_learned

            # --- velocity update ---
            if predictor_corrector or probability_flow:
                if exponential_integration:
                    vt_reverse = (torch.exp(dt) * vt_reverse
                                  + torch.expm1(dt) * scorev_reverse)
                else:
                    vt_reverse = vt_reverse - (-vt_reverse - scorev_reverse) * dt
            else:
                if exponential_integration:
                    vt_reverse = (torch.exp(dt) * vt_reverse
                                  + 2 * torch.expm1(dt) * scorev_reverse
                                  + torch.sqrt(torch.expm1(2 * dt)) * eps_v)
                else:
                    vt_reverse = (vt_reverse
                                  - (-vt_reverse - 2 * scorev_reverse) * dt
                                  + torch.sqrt(2 * dt) * eps_v)

            # --- position update and graph refresh ---
            ft_reverse = wrap_angle(ft_reverse - dt * vt_reverse)
            graph = self._update_graph_batch(graph, ft_reverse)

            # --- corrector ---
            if predictor_corrector and i < n_steps - 2:
                corr_step_size = tau * t_g_next ** 2 / self.total_time ** 2  # (num_graphs,1)
                for _ in range(predictor_corrector_n_steps):
                    if not only_correct_vt:
                        ft_reverse, vt_reverse = self._langevin_corrector_step_graph(
                            graph, vt_reverse, t_g, batch_vec, dt,
                            tdm_score_fn, tau=float(corr_step_size.mean().item()),
                        )
                        graph = self._update_graph_batch(graph, ft_reverse)
                    else:
                        vt_reverse = self._langevin_corrector_step_graph(
                            graph, vt_reverse, t_g, batch_vec, dt,
                            tdm_score_fn, tau=float(corr_step_size.mean().item()),
                            only_correct_vt=True,
                        )

            sigma_list.append(float(self.sde.sigma_t(t_g_next)[0, 0].item()))
            t_list.append(float(t_g_next[0, 0].item()))
            if sample_trajectory:
                ft_traj.append(ft_reverse.cpu())
                vt_traj.append(vt_reverse.cpu())

        t_arr = torch.tensor(t_list).numpy()
        sigma_arr = torch.tensor(sigma_list).numpy()

        if debug:
            return ft_traj, vt_traj, t_arr, sigma_arr
        if sample_trajectory:
            return ft_traj, vt_traj, t_arr
        return ft_reverse, vt_reverse

    def _langevin_corrector_step_graph(
        self,
        graph: Data,
        vt_reverse: torch.Tensor,
        t_g: torch.Tensor,
        batch_vec: torch.Tensor,
        dt: torch.Tensor,
        score_fn: Callable[[Data, torch.Tensor, torch.Tensor], torch.Tensor],
        tau: float = 0.25,
        only_correct_vt: bool = False,
    ):
        """Langevin corrector step operating on node-level tensors."""
        t_n = t_g[batch_vec]                                       # (N_total, 1)
        score_learned = score_fn(graph, vt_reverse, t_g)           # (N_total, 2)
        tau_t = torch.tensor(tau, device=vt_reverse.device, dtype=vt_reverse.dtype)
        eps_v = torch.randn_like(vt_reverse)
        prefactor = self._get_prefector(t_n)
        sigma_norm_r_t = torch.sqrt(self._sigma_norm_t(t_n))
        sigma_v = self.sde.sigma_t(t_n)
        scorev_reverse = (score_learned * prefactor * sigma_norm_r_t
                          - vt_reverse / (sigma_v ** 2))
        vt_reverse = vt_reverse + tau_t * scorev_reverse + torch.sqrt(2 * tau_t) * eps_v
        if not only_correct_vt:
            ft_reverse = wrap_angle(graph.x - dt * vt_reverse)
            return ft_reverse, vt_reverse
        return vt_reverse

    """
    Graph utilities
    """

    def _update_graph_batch(self, graph: Data, new_pos: torch.Tensor):
        # update the node features and positions        
        graph.x = new_pos
        graph.pos = new_pos
        diff = new_pos[graph.edge_index[0]] - new_pos[graph.edge_index[1]]
        # diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        graph.edge_attr = diff
        return graph

    def _center_graph_sample(self, graph: Data):
        # center the graph sample
        graph.x = graph.x - scatter(graph.x, graph.batch, dim=0, reduce="mean")
        return graph