"""
Train a joint lattice + fractional-coordinate score network (``CSPVNet``) on a
crystal dataset under the ``KLDM`` diffusion, wrapped in the ``LitCSPVNet``
LightningModule.

This configuration does NOT predict atom types ``h`` (``pred_h=False``): only
the lattice score (``l``) and the fractional-coordinate score (``v``) are
learned. Atom types are treated as given conditioning inputs.
"""

import os
from datetime import datetime

import torch
import torch_geometric.transforms as PyGT
from torch_geometric.loader import DataLoader as PyGDataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

import src.dataLib.transform as T
from src.dataLib.realCrystal import CrystalDataset
from src.nn.CSPVNet import CSPVNet
from src.diffusion.tdm import TDMDiffusion
from src.diffusion.continuous import ContinuousDiffusion
from src.diffusion.kldm import KLDM
from src.diffusion.sde import VPSDE, LinearSchedule
from src.lit.litCSPVNet import LitCSPVNet
from src.device import get_default_device, get_lightning_accelerator


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
total_time = 2.0
dim = 3
n_epoch = 500
lr = 1e-3
batch_size = 256

data_folder = "data/carbon-24"
dataset_name = "carbon24"

diffusion_kwargs = {
    "t_dist_kw": "uniform",
    "t_min": 5e-3,
    "shared_time": True,  # single per-graph time shared by lattice + coords
}

nn_kwargs = {
    "hidden_dim": 256,
    "time_dim": 128,
    "num_layers": 6,
    "h_dim": 100,       # embedding table size for atom types (indexed by atomic number)
    "num_freqs": 10,
    "ln": True,
    "smooth": False,    # embed discrete atom types via nn.Embedding
    "pred_h": False,    # <-- do not predict atom types
    "pred_l": True,
    "pred_v": True,
    "zero_cog": True,
}

lit_kwargs = {
    "lambda_l": 1.0,
    "lambda_f": 1.0,
}


def build_transform() -> PyGT.Compose:
    return PyGT.Compose(
        [
            T.ContinuousIntervalLengths(in_key="lengths", out_key="lengths"),
            T.ContinuousIntervalAngles(in_key="angles", out_key="angles"),
            T.FullyConnectedGraph(key="edge_index", len_from="pos"),
            # lattice vector l = (log a, log b, log c, tan alpha, tan beta, tan gamma)
            T.ConcatFeatures(in_keys=["lengths", "angles"], out_key="l"),
            T.Pos2Angle(in_key="pos", out_key="pos"),
            # crystal data is stored in float64; match float32 network weights
            T.CastFloating(dtype=torch.float32),
            
        ]
    )


def build_model() -> CSPVNet:
    return CSPVNet(
        hidden_dim=nn_kwargs["hidden_dim"],
        time_dim=nn_kwargs["time_dim"],
        num_layers=nn_kwargs["num_layers"],
        h_dim=nn_kwargs["h_dim"],
        num_freqs=nn_kwargs["num_freqs"],
        ln=nn_kwargs["ln"],
        smooth=nn_kwargs["smooth"],
        pred_h=nn_kwargs["pred_h"],
        pred_l=nn_kwargs["pred_l"],
        pred_v=nn_kwargs["pred_v"],
        zero_cog=nn_kwargs["zero_cog"],
    )


def build_kldm() -> KLDM:
    diffusion_f = TDMDiffusion(
        dim=dim,
        integrator_type="Euler",
        trunc_n=13,
        f_scale=2 * torch.pi,
        total_time=total_time,
        simplified_param=True,
        n_sigma_rs=2000,
        zero_cog=nn_kwargs["zero_cog"],
    )
    l_sde = VPSDE(schedule=LinearSchedule(beta_min=2.0, beta_max=2.0))
    diffusion_l = ContinuousDiffusion(sde=l_sde, total_time=total_time, parameterization="x0")
    return KLDM(
        l_diffusion=diffusion_l,
        f_diffusion=diffusion_f,
        h_diffusion=None,
        total_time=total_time,
    )


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = get_default_device()
    accelerator = get_lightning_accelerator(device)

    transform = build_transform()
    train_ds = CrystalDataset(path=os.path.join(data_folder, "train.pt"), transform=transform)
    val_ds = CrystalDataset(path=os.path.join(data_folder, "val.pt"), transform=transform)

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model()
    kldm = build_kldm()

    lit_model = LitCSPVNet(
        model=model,
        kldm=kldm,
        diffusion_kwargs=diffusion_kwargs,
        batch_size=batch_size,
        lr=lr,
        lambda_l=lit_kwargs["lambda_l"],
        lambda_f=lit_kwargs["lambda_f"],
    )

    experiment_name = f"CSPVNet_KLDM_{dataset_name}_no_h"
    wandb_logger = WandbLogger(
        name=experiment_name,
        save_dir="wandb_logs",
        project="diffusion-playground",
        checkpoint_name=experiment_name,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{timestamp}/{experiment_name}",
        filename="cspvnet_{epoch:02d}-{validation/loss:.4f}",
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=n_epoch,
        accelerator=accelerator,
        log_every_n_steps=32,
        gradient_clip_val=1.0,  # match the hand-written loops; prevents score-matching blow-ups
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
