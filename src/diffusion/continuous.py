from src.diffusion.base import BaseDiffusion
from src.diffusion.sde import BaseSDE, VPSDE, EulerIntegrator
import torch
import torch.nn as nn
from typing import Literal, Callable, Optional


class ContinuousDiffusion(BaseDiffusion):
    def __init__(
        self, 
        sde: BaseSDE,
        total_time: float = 2.0,
        parameterization: Literal["noise", "x0", "score"] = "score",
    ):
        super().__init__()
        self.sde = sde
        self.integrator = EulerIntegrator(sde=sde)

    def sample_forward(
        self,
        x0: torch.Tensor,  # shape (batch_size, num_features, dim), because the data is not graph data, so it is not a tensor of shape (batch_size, dim)
        t_dist_kw: Literal["uniform", "quadratic", "linear", "constant"]="uniform", 
        n_steps: int = 100, 
        constant_t: float = 1.0,
        return_time: bool = False,
        t_min: float = 5e-3,
    ):
        device = x0.device
        dtype = x0.dtype
        batch_size = x0.shape[0]
        dim = x0.shape[1]

        # sample time uniformly
        ts = torch.rand(size=(batch_size,1), device=device, dtype=dtype) # shape (batch_size, 1)
        if t_dist_kw == "uniform":
            ts = ts * (self.total_time - t_min) + t_min
        elif t_dist_kw == "quadratic":
            ts = ts ** 2
        elif t_dist_kw == "linear":
            ts = torch.linspace(0, self.total_time, n_steps, device=device, dtype=dtype) # shape (n_steps)
            ts = torch.unsqueeze(ts,dim=0).repeat(batch_size,1) # shape (batch_size, n_steps)
        elif t_dist_kw == "constant":
            ts = constant_t * torch.ones(size=(batch_size,1), device=device, dtype=dtype)
        expanded_ts = self._expand_time_to_x(ts, x0)
        # sample the noised data
        xt = self._sample_xt(x0, expanded_ts)
        # compute the score of the noised data
        target = self._compute_target(x0, xt, expanded_ts)

        # return the noised data and the score
        if return_time:
            return xt, target, ts
        else:
            return xt, target

    # helper function to sample the noised data
    def _sample_xt(self, x0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        if ts.shape[0] != x0.shape[0]:
            raise ValueError(
                f"Batch mismatch: x0 has batch {x0.shape[0]}, ts has batch {ts.shape[0]}"
            )

        mean_t_coeff = self.sde.mean_t_coeff(ts) # shape (batch_size, 1, ..., 1)  in this case, mean_t_coeff has shape (batch_size, 1, 1)
        sigma_t = self.sde.sigma_t(ts) # shape (batch_size, 1, ..., 1) in this case, sigma_t has shape (batch_size, 1, 1)
        eps = torch.randn_like(x0) # shape (batch_size, num_features, dim)
        xt = mean_t_coeff * x0 + sigma_t * eps # shape (batch_size, num_features, dim)
        return xt # shape (batch_size, num_features, dim)

    def _compute_score(self, x0: torch.Tensor, xt: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        if x0.shape != xt.shape:
            raise ValueError(
                f"x0 and xt must have the same shape, got shape {x0.shape} and {xt.shape}"
            )
        if x0.shape[0] != xt.shape[0]:
            raise ValueError(
                f"Batch mismatch: x0 has batch {x0.shape[0]}, xt has batch {xt.shape[0]}"
            )
        sigma_t = self.sde.sigma_t(ts) # shape (batch_size, 1, ..., 1) in this case, sigma_t has shape (batch_size, 1, 1)
        score = -(xt - x0) / (sigma_t**2)
        return score # shape (batch_size, num_features, dim)
    
    def _compute_target(self, x0: torch.Tensor, xt: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        if x0.shape != xt.shape:
            raise ValueError(
                f"x0 and xt must have the same shape, got shape {x0.shape} and {xt.shape}"
            )
        if x0.shape[0] != xt.shape[0]:
            raise ValueError(
                f"Batch mismatch: x0 has batch {x0.shape[0]}, xt has batch {xt.shape[0]}"
            )
        if self.parameterization == "noise":
            raise NotImplementedError
        elif self.parameterization == "x0":
            raise NotImplementedError
        elif self.parameterization == "score":
            target = self._compute_score(x0, xt, ts)
        else:
            raise ValueError(
                f"Invalid parameterization: {self.parameterization}"
            )
        return target

    def _expand_time_to_x(self, ts: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Accept ts in shape (B,) or (B, 1), then expand to (B, 1, ..., 1)
        # so it broadcasts over all non-batch dimensions of x0.
        if ts.ndim == 1:
            ts = ts.unsqueeze(-1)
        if ts.ndim > 1 and ts.shape[1] != 1:
            raise ValueError(
                f"ts must provide one time value per sample, got shape {tuple(ts.shape)}"
            )
        ts = ts.reshape(ts.shape[0], *([1] * (x.ndim - 1))) # shape (batch_size, 1, ..., 1) same number of dimensions as x
        return ts

    """
    Inference step throw the time reversed diffusion,
    This is contains all the steps of the time reversed diffusion, including the langevin corrector step and the score network step.
    """
    @torch.inference_mode()
    def sample_backward(
        self,
        xT: torch.Tensor,
        marginal_score_fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor], 
        n_steps: int = 100,
    ):
        device = xT.device
        dtype = xT.dtype
        batch_size = xT.shape[0]

        raise NotImplementedError

    """
    Sample one step of the time reversed diffusion,
    This is the basic step of the time reversed diffusion, including the langevin corrector step and the score network step.
    """
    def backward_step_euler(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        dt: torch.Tensor,
        probability_flow: bool = False,
    ):
        dW = torch.sqrt(dt)*torch.randn_like(xt)
        if probability_flow:
            xt_next = xt + self.sde.reverse_drift(xt, t, score,eta=0) * (-dt) + self.sde.diffusion(xt,t) * dW # probability flow ODE
        else:
            xt_next = xt + self.sde.reverse_drift(xt, t, score) * (-dt) + self.sde.diffusion(xt,t) * dW
        return xt_next


    def langevin_corrector_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        dt: torch.Tensor,
        tau: float = 0.25,
    ):
        eps = torch.randn_like(xt)
        xt_corrected = xt + tau * score * dt + torch.sqrt(2 * tau) * eps # Langevin corrector step
        return xt_corrected
    
    def tweedie_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ):  
        alpha_t = self.sde.mean_t_coeff(t)
        sigma_t = self.sde.sigma_t(t)
        x0_hat = (xt + score * sigma_t**2) / alpha_t
        return x0_hat

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ):
        assert pred.shape == target.shape
        loss = torch.nn.functional.mse_loss(pred, target) # mean squared error loss
        return loss