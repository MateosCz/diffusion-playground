import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import lightning as L
from torch_geometric.data import Data
from src.diffusion import TDMDiffusion  
from src.trainGraph import weighted_score_loss

class LitVanillaGNN(L.LightningModule):
    def __init__(self, model: nn.Module, diffusion: TDMDiffusion, diffusion_kwargs: dict, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.diffusion = diffusion
        self.diffusion_kwargs = diffusion_kwargs

    def training_step(self, batch: Data) -> torch.Tensor:
        batch_graph = batch
        (v_t, batch_graph_noised), target_score, t_scalar = self.diffusion.sample_forward_graph(
            graph=batch_graph,
            t_dist_kw=self.diffusion_kwargs["t_dist_kw"],
            v0_dist_kw=self.diffusion_kwargs["v0_dist_kw"],
            return_time=True,
            zero_cog=self.diffusion_kwargs["zero_cog"]
        )
        pred_score = self.model.forward_from_data(batch_graph_noised, v_t, t_scalar)
        t_all = t_scalar[batch_graph.batch]
        loss = weighted_score_loss(pred_score, target_score, t_all, self.diffusion.total_time)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer