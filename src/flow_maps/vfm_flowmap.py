from src.flow_matching.rg_vfm import RGVFM
import torch
import numpy as np
import torch.nn as nn


class VFMFlowMap:
    """
    Now only implement the extrinsic version.
    """
    def __init__(self, rg_vfm: RGVFM):
        self.rg_vfm = rg_vfm

    def Xst(self, x_s, s, t, pi: nn.Module):
        """Affine flow map: x_s + (t-s) * (pi_s,t(x_s) - x_s) / (T-s)."""
        s = self.rg_vfm.batch_time(s, x_s)
        t = self.rg_vfm.batch_time(t, x_s)
        if torch.any(t < s):
            raise ValueError("t must be greater than or equal to s")


        return x_s + (t - s) * (pi(x_s, s, t) - x_s) / (self.rg_vfm.total_time - s)

    def tangent_loss(self,x1, xt, t, pi: nn.Module):
        """tangent loss for the flow map
        E[||(pi_t,t(x_t) - x_1)/(1-t)||^2]
        """
        return self.rg_vfm.loss(pi(xt, t, t), x1, t=t)
    
    def lagrange_loss(self, xs, s, t, pi: nn.Module):
        """
        velocity parameterization:E[||partial\_t X\_s,t(x\_s) - v\_t,t(X\_s,t(x\_s))||^2]  = 
        endpoint parameterization:E[||partial_t X_s,t(x_s) - (sg(pi_t,t(X_s,t(x_s))) - X_s,t(x_s))/(1-t)||^2]
        Stop-gradient applies only to pi_t,t; X_s,t remains differentiable.
        """
        s = self.rg_vfm.batch_time(s, xs)
        t = self.rg_vfm.batch_time(t, xs)

        x_st, dx_dt = torch.autograd.functional.jvp(
            lambda target_t: self.Xst(xs, s, target_t, pi),
            (t,),
            (torch.ones_like(t),),
            create_graph=True,
        )
        v_tt_fake = (pi(x_st, t, t).detach() - x_st)
        return (dx_dt * (self.rg_vfm.total_time - t) - v_tt_fake).square().sum(dim=-1).mean()

    @torch.inference_mode()
    def sample(self, pi: nn.Module, n_steps: int, num_samples: int):
        """Sample from the trained flow-map model by applying n_steps flow-map updates from 0 to total_time.

        Return continuous model-space states of shape (num_samples, model_dim).
        """
        reference = next(pi.parameters())
        time_steps = torch.linspace(
            0, self.rg_vfm.total_time, n_steps + 1,
            device=reference.device, dtype=reference.dtype,
        )
        x = self.rg_vfm.sample_prior(
            (num_samples, self.rg_vfm.model_dim),
            device=reference.device,
            dtype=reference.dtype,
        )
        for s, t in zip(time_steps[:-1], time_steps[1:]):
            x = self.Xst(x, s, t, pi)
        return x
