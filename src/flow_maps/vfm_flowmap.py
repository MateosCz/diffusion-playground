from src.flow_matching.rg_vfm import RGVFM
from src.flow_matching.vfm import VFM
import torch
import torch.nn as nn


class VFMFlowMap:
    """Affine endpoint flow map for Euclidean or ambient-space states."""

    def __init__(self, rg_vfm: RGVFM | VFM, is_graph_data: bool = False):
        self.vfm = rg_vfm
        # Keep existing fractional-coordinate callers working.
        self.rg_vfm = rg_vfm
        self.is_graph_data = is_graph_data
        
    def Xst(self, x_s, s, t, pi: nn.Module):
        """Affine flow map: x_s + (t-s) * (pi_s,t(x_s) - x_s) / (T-s)."""
        s = self.vfm.batch_time(s, x_s)
        t = self.vfm.batch_time(t, x_s)
        if torch.any(t < s):
            raise ValueError("t must be greater than or equal to s")


        return x_s + (t - s) * (pi(x_s, s, t) - x_s) / (self.vfm.total_time - s)

    def tangent_loss(self,x1, xt, t, pi: nn.Module):
        """Batch-averaged endpoint regression loss at s=t."""
        return self.vfm.loss(pi(xt, t, t), x1, t=t).mean()
    
    def lagrange_loss(self, xs, s, t, pi: nn.Module):
        """Match (T-t) * dX_st/dt to stop_grad(pi_tt(X_st)) - X_st.

        Stop-gradient applies only to pi_tt; X_st remains differentiable.
        """
        s = self.vfm.batch_time(s, xs)
        t = self.vfm.batch_time(t, xs)

        x_st, dx_dt = torch.autograd.functional.jvp(
            lambda target_t: self.Xst(xs, s, target_t, pi),
            (t,),
            (torch.ones_like(t),),
            create_graph=True,
        )
        v_tt_fake = (pi(x_st, t, t) - x_st).detach()
        return (dx_dt * (self.vfm.total_time - t) - v_tt_fake).square().sum(dim=-1).mean()

    @torch.inference_mode()
    def sample(self, pi: nn.Module, n_steps: int, num_samples: int):
        """Sample from the trained flow-map model by applying n_steps flow-map updates from 0 to total_time.

        Return continuous model-space states of shape (num_samples, model_dim).
        """
        reference = next(pi.parameters())
        time_steps = torch.linspace(
            0, self.vfm.total_time, n_steps + 1,
            device=reference.device, dtype=reference.dtype,
        )
        x = self.vfm.sample_prior(
            (num_samples, self.vfm.model_dim),
            device=reference.device,
            dtype=reference.dtype,
        )
        for s, t in zip(time_steps[:-1], time_steps[1:]):
            x = self.Xst(x, s, t, pi)
        return x

__all__ = ["VFMFlowMap"]
