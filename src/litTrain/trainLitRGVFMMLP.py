"""Train ``RGVFMMLP`` on 1D or 2D fractional-coordinate torus data."""

from datetime import datetime

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from src.dataLib.synthetic import (
    Checkerboard_Dataset,
    Pacman_Dataset,
)
from src.device import get_default_device, get_lightning_accelerator
from src.flow_matching import RGVFM
from src.lit.checkerboard_generation_metrics import CheckerboardGenerationMetrics
from src.lit.callbacks import last_checkpoint
from src.lit.litRGVFMMLP import LitRGVFMMLP
from src.manifolds import FlatTorus01
from src.nn.rg_vfm_mlp import EX_RGVFMMLP, RGVFMMLP


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
total_time = 1.0
dim = 1  # Use 2 for the 2D checkerboard or pacman.
n_epoch = 200
lr = 1e-4
batch_size = 512
num_workers = 6
num_rows = 8

dataset_name = "checkerboard"  # "checkerboard" or "pacman"
if dim == 2 and dataset_name == "checkerboard":
    dataset_name = f"checkerboard_{num_rows}x{num_rows}"
elif dim == 1 and dataset_name == "checkerboard":
    dataset_name = f"checkerboard_{num_rows}"

pacman_path = "data/pacman.npy"
train_size = 40_000
val_size = 4_096

flow_kwargs = {
    "time_distribution": "uniform",
    "t_min": 0.0,
    "constant_time": 0.5,
}

rg_vfm_kwargs = {
    "total_time": total_time,
    "time_eps": 1e-3,
    "noise_scale": 0.0,
    "max_velocity_scale": None,
    "max_loss_weight": 100.0,
    "normalize_loss_weights": False, # True or False
    "normalize_loss": False,
    "support": "extrinsic",  # "intrinsic" or "extrinsic"
    "intrinsic_prior_std": 1.0,
    "integrator": "euler",
    "is_loss_weighted": True,
    "ambient_metric": "euclidean",
    # "ambient_metric": "geodesic",
}

nn_kwargs = {
    "dim": dim,
    "x_lifting_dim": 256,
    "time_embedding_half_dim": 128,
    "hidden_dim": [512,1024, 512],
    "output_dim": dim,
    "total_time": total_time,
    "time_embedding_scale": 1.0,
    "position_fourier_bands": 8,
    # Raw coordinates create an artificial discontinuity at the 0/1 seam.
    "with_residual_position": True,
    "with_sincos_position": True,
    "residual_position_scale": 0.01,
}

generation_eval_every_n_epochs = 25
generation_eval_samples = 4_096
generation_eval_steps = 100

# Extrinsic geometry and dimensions are selected automatically. Override
# residual prediction here independently of the intrinsic configuration.
# Ambient coordinates are Cartesian, so keep their raw values.
extrinsic_nn_kwargs = {"with_residual_position": False, "with_sincos_position": False, "project_to_manifold": True}


def build_dataset(name: str, size: int, *, seed: int | None = None) -> Dataset:
    """Create fractional-coordinate data directly in ``[0, 1)``."""
    if name.startswith("checkerboard"):
        base_dataset = Checkerboard_Dataset(
            num_rows=num_rows,
            dataset_size=size,
            seed=seed,
            dim=dim,
        )
    elif name == "pacman":
        base_dataset = Pacman_Dataset(
            directory=pacman_path,
            size=size,
            seed=seed,
        )
    else:
        raise ValueError(
            f"dataset_name must start with 'checkerboard' or equal 'pacman', got {name!r}"
        )
    return base_dataset


def build_loaders() -> tuple[DataLoader, DataLoader]:
    train_dataset = build_dataset(dataset_name, train_size)
    val_dataset = build_dataset(dataset_name, val_size, seed=10_000)
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,**common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True,**common_kwargs)
    return train_loader, val_loader


def build_manifold(manifold_dim: int = dim) -> FlatTorus01:
    """Build the canonical ``[0, 1)`` torus for fractional coordinates."""
    return FlatTorus01(dim=manifold_dim)


def build_model(manifold: FlatTorus01 | None = None) -> RGVFMMLP:
    manifold = manifold or build_manifold()
    model_class = EX_RGVFMMLP if rg_vfm_kwargs["support"] == "extrinsic" else RGVFMMLP
    return model_class(
        **build_nn_kwargs(manifold),
        manifold=manifold,
    )


def build_nn_kwargs(manifold: FlatTorus01) -> dict:
    kwargs = dict(nn_kwargs)
    if rg_vfm_kwargs["support"] == "extrinsic":
        kwargs.update(extrinsic_nn_kwargs)
        kwargs.update(
            dim=manifold.ambient_dim,
            output_dim=manifold.ambient_dim,
            )
    return kwargs


def build_rg_vfm(manifold: FlatTorus01 | None = None) -> RGVFM:
    manifold = manifold or build_manifold()
    return RGVFM(manifold, **rg_vfm_kwargs)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = get_default_device()
    accelerator = get_lightning_accelerator(device)
    train_loader, val_loader = build_loaders()
    manifold = build_manifold()

    model_kwargs = build_nn_kwargs(manifold)

    normalize_loss_weights_flag = 'normalized_loss_weight' if rg_vfm_kwargs['normalize_loss_weights'] else 'unnormalized_loss_weight'
    loss_weighted_flag = normalize_loss_weights_flag if rg_vfm_kwargs['is_loss_weighted'] else 'unweighted_loss'
    position_coding_flag = 'sincos' if model_kwargs['with_sincos_position'] else 'raw'
    ambient_distance_loss_flag = (
        f"{rg_vfm_kwargs['ambient_metric']}_loss"
        if rg_vfm_kwargs['support'] == 'extrinsic' else ''
    )
    experiment_name = f"RGVFMMLP_{dataset_name}_fractional_{rg_vfm_kwargs['support']}_{position_coding_flag}_{'no_res' if not model_kwargs['with_residual_position'] else 'res_scale_' + str(model_kwargs['residual_position_scale'])}_{loss_weighted_flag}_{ambient_distance_loss_flag}"
    checkpoint_dir = f"checkpoints/{timestamp}/{experiment_name}"
    lit_model = LitRGVFMMLP(
        model=build_model(manifold),
        rg_vfm=build_rg_vfm(manifold),
        flow_kwargs=flow_kwargs,
        rg_vfm_kwargs=rg_vfm_kwargs,
        nn_kwargs=model_kwargs,
        experiment_name_timestamp=experiment_name+"_"+timestamp,
        batch_size=batch_size,
        lr=lr,
    )
    wandb_logger = WandbLogger(
        name=experiment_name,
        save_dir="wandb_logs",
        project="diffusion-playground",
        checkpoint_name=experiment_name,
    )
    loss_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="rgvfm_mlp_{epoch:04d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    callbacks: list[L.Callback] = [
        loss_checkpoint,
        last_checkpoint(checkpoint_dir),
    ]
    if dataset_name.startswith("checkerboard"):
        callbacks.extend(
            [
                CheckerboardGenerationMetrics(
                    num_rows=num_rows,
                    bins=4 * num_rows,
                    n_samples=generation_eval_samples,
                    n_steps=generation_eval_steps,
                    every_n_epochs=generation_eval_every_n_epochs,
                ),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=(
                        "rgvfm_mlp_distribution_"
                        "{epoch:04d}-{val_generated_tv:.6f}"
                    ),
                    monitor="val_generated_tv",
                    mode="min",
                    save_top_k=1,
                    every_n_epochs=generation_eval_every_n_epochs,
                    auto_insert_metric_name=False,
                ),
            ]
        )

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=n_epoch,
        accelerator=accelerator,
        log_every_n_steps=32,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )

    try:
        trainer.fit(
            model=lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
