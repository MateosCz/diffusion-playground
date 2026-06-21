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
    def __init__(self, model: nn.Module, diffusion: TDMDiffusion, diffusion_kwargs: dict, batch_size: int, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.diffusion = diffusion
        self.diffusion_kwargs = diffusion_kwargs
        self._train_loss = []
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward_from_data(self, graph, vt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model.forward_from_data(graph, vt, t)


    def training_step(self, batch: Data) -> torch.Tensor:
        batch_graph = batch
        (v_t, batch_graph_noised), target_score, t_scalar = self.diffusion.sample_forward_graph(
            graph=batch_graph,
            t_dist_kw=self.diffusion_kwargs["t_dist_kw"],
            v0_dist_kw=self.diffusion_kwargs["v0_dist_kw"],
            return_time=True,
        )
        pred_score = self.model.forward_from_data(batch_graph_noised, v_t, t_scalar)
        t_all = t_scalar[batch_graph.batch]
        loss = weighted_score_loss(pred_score, target_score, t_all, self.diffusion.total_time)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=self.batch_size)
        self._train_loss.append(loss.detach())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch: Data) -> torch.Tensor:
        batch_graph = batch
        (v_t, batch_graph_noised), target_score, t_scalar = self.diffusion.sample_forward_graph(
            graph=batch_graph,
            t_dist_kw=self.diffusion_kwargs["t_dist_kw"],
            v0_dist_kw=self.diffusion_kwargs["v0_dist_kw"],
            return_time=True,
        )
        pred_score = self.model.forward_from_data(batch_graph_noised, v_t, t_scalar)
        t_all = t_scalar[batch_graph.batch]
        loss = weighted_score_loss(pred_score, target_score, t_all, self.diffusion.total_time)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size = self.batch_size)
        return loss

    def validation_step(self, batch: Data):
        batch_graph = batch
        (v_t, batch_graph_noised), target_score, t_scalar = self.diffusion.sample_forward_graph(
            graph=batch_graph,
            t_dist_kw=self.diffusion_kwargs["t_dist_kw"],
            v0_dist_kw=self.diffusion_kwargs["v0_dist_kw"],
            return_time=True,
        )
        pred_score = self.model.forward_from_data(batch_graph_noised, v_t, t_scalar)
        t_all = t_scalar[batch_graph.batch]
        loss = weighted_score_loss(pred_score, target_score, t_all, self.diffusion.total_time)
        self.log("validation/loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size = self.batch_size)
        return loss

    def on_train_epoch_end(self):
        losses = torch.stack(self._train_loss)
        mean_loss = losses.mean()
        var = losses.var(unbiased=False)
        std = losses.std(unbiased=False)
        self.log_dict({
            "train/loss_mean": mean_loss,
            "train/loss_var": var,
            "train/loss_std": std
            
        })
        self._train_loss.clear()
    