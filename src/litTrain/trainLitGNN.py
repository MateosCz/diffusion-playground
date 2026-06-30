from src.lit.litGNN import LitVanillaGNN
from src.nn.vanillaGNN import TDM_VanillaGNN
from src.diffusion import TDMDiffusion
from src.trainGraph import weighted_score_loss
from src.dataLib.synthetic import Shapes_Dataset, PyGGraphWrapper
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from datetime import datetime
from src.device import get_default_device, get_lightning_accelerator

pt_invariant = True
total_time = 2.0
dim = 2
n_epoch = 400
lr = 1e-3
base_ds_kw = "triangle"
# base_ds_kw = "mix"

device = get_default_device()
accelerator = get_lightning_accelerator(device)
diffusion_kwargs_no_zero_cog = {
    "t_dist_kw": "uniform",
    "v0_dist_kw": "zero",
    "zero_cog": False,
    "zero_cog_score": False
}

diffusion_kwargs_zero_cog_score = {
    "t_dist_kw": "uniform",
    "v0_dist_kw": "zero",
    "zero_cog": pt_invariant,
    "zero_cog_score": True
}
diffusion_kwargs_no_zero_cog_score = {
    "t_dist_kw": "uniform",
    "v0_dist_kw": "zero",
    "zero_cog": pt_invariant,
    "zero_cog_score": False
}

nn_kwargs = {
    "node_feat_dim": 2,
    "edge_fourier_bands": 8,
    "v_dim": 2,
    "hidden_dim": [512,512],
    "num_mp_layers": 6,
    "time_embedding_half_dim": 32,
    "output_dim": 2,
    "total_time": total_time,
    "time_embedding_scale": 1.0,
    "with_sincos_position": True,
    "only_sincos_position": True,
    "position_fourier_bands": 8,
    "pt_invariant": pt_invariant,
    "zero_cog": pt_invariant
}

data_kwargs = {
    "num_points": 32,
    "dataset_size": 1000,
    "shape_types": ["triangle"] if base_ds_kw == "triangle" else ["triangle", "rectangle", "star"],
    "centered": pt_invariant,
    "num_points_range": (26,32),
    "batch_size": 32,
    "fix_rotation": True
}





def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_ds = Shapes_Dataset(
        num_points=data_kwargs["num_points"],
        dataset_size=data_kwargs["dataset_size"],
        shape_types=data_kwargs["shape_types"],
        centered=data_kwargs["centered"],
        fix_rotation=data_kwargs["fix_rotation"]
    )

    graph_ds = PyGGraphWrapper(
        base_ds, 
        num_points_range=data_kwargs["num_points_range"])
    
    loader = PyGDataLoader(
        graph_ds,
        batch_size=data_kwargs["batch_size"],
        shuffle=True
    )
    diffusion_no_zero_cog = TDMDiffusion(
        dim=dim,
        integrator_type="Euler",
        simplified_param=True,
        zero_cog=nn_kwargs["zero_cog"],
        zero_cog_score=diffusion_kwargs_no_zero_cog["zero_cog_score"]
    )

    diffusion_zero_cog_score = TDMDiffusion(
        dim=dim,
        integrator_type="Euler",
        simplified_param=True,
        zero_cog=nn_kwargs["zero_cog"],
        zero_cog_score=diffusion_kwargs_zero_cog_score["zero_cog_score"]
    )
    diffusion_no_zero_cog_score = TDMDiffusion(
        dim=dim,
        integrator_type="Euler",
        simplified_param=True,
        zero_cog=nn_kwargs["zero_cog"],
        zero_cog_score=diffusion_kwargs_no_zero_cog_score["zero_cog_score"]
    )

    gnn_no_zero_cog = TDM_VanillaGNN(
        node_feat_dim=nn_kwargs["node_feat_dim"],
        edge_fourier_bands=nn_kwargs["edge_fourier_bands"],
        v_dim=nn_kwargs["v_dim"],
        hidden_dim=nn_kwargs["hidden_dim"],
        num_mp_layers=nn_kwargs["num_mp_layers"],
        time_embedding_half_dim=nn_kwargs["time_embedding_half_dim"],
        output_dim=nn_kwargs["output_dim"],
        total_time=nn_kwargs["total_time"],
        time_embedding_scale=nn_kwargs["time_embedding_scale"],
        with_sincos_position=nn_kwargs["with_sincos_position"],
        only_sincos_position=nn_kwargs["only_sincos_position"],
        position_fourier_bands=nn_kwargs["position_fourier_bands"],
        pt_invariant=nn_kwargs["pt_invariant"],
        zero_cog=nn_kwargs["zero_cog"],
        zero_cog_score=diffusion_kwargs_no_zero_cog["zero_cog_score"]
    )

    gnn_zero_cog_score = TDM_VanillaGNN(
        node_feat_dim=nn_kwargs["node_feat_dim"],
        edge_fourier_bands=nn_kwargs["edge_fourier_bands"],
        v_dim=nn_kwargs["v_dim"],
        hidden_dim=nn_kwargs["hidden_dim"],
        num_mp_layers=nn_kwargs["num_mp_layers"],
        time_embedding_half_dim=nn_kwargs["time_embedding_half_dim"],
        output_dim=nn_kwargs["output_dim"],
        total_time=nn_kwargs["total_time"],
        time_embedding_scale=nn_kwargs["time_embedding_scale"],
        with_sincos_position=nn_kwargs["with_sincos_position"],
        only_sincos_position=nn_kwargs["only_sincos_position"],
        position_fourier_bands=nn_kwargs["position_fourier_bands"],
        pt_invariant=nn_kwargs["pt_invariant"],
        zero_cog=nn_kwargs["zero_cog"],
        zero_cog_score=diffusion_kwargs_zero_cog_score["zero_cog_score"]
    )

    gnn_no_zero_cog_score = TDM_VanillaGNN(
        node_feat_dim=nn_kwargs["node_feat_dim"],
        edge_fourier_bands=nn_kwargs["edge_fourier_bands"],
        v_dim=nn_kwargs["v_dim"],
        hidden_dim=nn_kwargs["hidden_dim"],
        num_mp_layers=nn_kwargs["num_mp_layers"],
        time_embedding_half_dim=nn_kwargs["time_embedding_half_dim"],
        output_dim=nn_kwargs["output_dim"],
        total_time=nn_kwargs["total_time"],
        time_embedding_scale=nn_kwargs["time_embedding_scale"],
        with_sincos_position=nn_kwargs["with_sincos_position"],
        only_sincos_position=nn_kwargs["only_sincos_position"],
        position_fourier_bands=nn_kwargs["position_fourier_bands"],
        pt_invariant=nn_kwargs["pt_invariant"],
        zero_cog=nn_kwargs["zero_cog"],
        zero_cog_score=diffusion_kwargs_no_zero_cog_score["zero_cog_score"]
    )



    # ------- run lit model on WandB


    lit_gnn_no_zero_cog = LitVanillaGNN(
        model=gnn_no_zero_cog,
        diffusion=diffusion_no_zero_cog,
        diffusion_kwargs=diffusion_kwargs_no_zero_cog,
        batch_size = data_kwargs["batch_size"],
        lr=lr
    )

    experiment_name_no_zero_cog = f"ptiGNN_shapes_{base_ds_kw}_no_zero_cog" if pt_invariant else f"GNN_shapes_{base_ds_kw}_no_zero_cog"
    wandb_logger_no_zero_cog = WandbLogger(
        name=experiment_name_no_zero_cog,
        save_dir="wandb_logs",
        project = "diffusion-playground",
        checkpoint_name = "nozerocog"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath = f"checkpoints/{timestamp}/{experiment_name_no_zero_cog}",
        filename = "nozerocog_{epoch:02d}-{val_loss:.4f}",
        monitor = "val_loss",
        mode = "min",
        save_top_k = 1,
        save_last = True
    )

    trainer_no_zero_cog = L.Trainer(
        logger=wandb_logger_no_zero_cog,
        max_epochs=n_epoch,
        accelerator=accelerator,
        log_every_n_steps=32,
        callbacks = [checkpoint_callback]
    )

    trainer_no_zero_cog.fit(model=lit_gnn_no_zero_cog, train_dataloaders=loader)
    wandb.finish()

    lit_gnn_zero_cog_score = LitVanillaGNN(
        model=gnn_zero_cog_score,
        diffusion=diffusion_zero_cog_score,
        diffusion_kwargs=diffusion_kwargs_zero_cog_score,
        batch_size = data_kwargs["batch_size"],
        lr=lr
    )
    
    experiment_name_zero_cog_score = f"ptiGNN_shapes_{base_ds_kw}_zero_cog_score" if pt_invariant else f"GNN_shapes_{base_ds_kw}_zero_cog_score"
    wandb_logger_zero_cog_score = WandbLogger(
        name=experiment_name_zero_cog_score,
        save_dir="wandb_logs",
        project = "diffusion-playground",
        checkpoint_name = "zerocogscore"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath = f"checkpoints/{timestamp}/{experiment_name_zero_cog_score}",
        filename = "zerocogscore_{epoch:02d}-{val_loss:.4f}",
        monitor = "val_loss",
        mode = "min",
        save_top_k = 1,
        save_last = True
    )   
    
    trainer_zero_cog_score = L.Trainer(
        logger=wandb_logger_zero_cog_score,
        max_epochs=n_epoch,
        accelerator=accelerator,
        log_every_n_steps=32,
        callbacks = [checkpoint_callback]
    )



    trainer_zero_cog_score.fit(model=lit_gnn_zero_cog_score, train_dataloaders=loader)
    wandb.finish()

    experiment_name_no_zero_cog_score = f"ptiGNN_shapes_{base_ds_kw}_no_zero_cog_score" if pt_invariant else f"GNN_shapes_{base_ds_kw}_no_zero_cog_score"

    lit_gnn_no_zero_cog_score = LitVanillaGNN(
        model=gnn_no_zero_cog_score,
        diffusion=diffusion_no_zero_cog_score,
        diffusion_kwargs=diffusion_kwargs_no_zero_cog_score,
        batch_size = data_kwargs["batch_size"],
        lr=lr
    )
    wandb_logger_no_zero_cog_score = WandbLogger(
        name=experiment_name_no_zero_cog_score,
        save_dir="wandb_logs",
        project = "diffusion-playground",
        checkpoint_name = "zerocog_nozerocogscore"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath = f"checkpoints/{timestamp}/{experiment_name_no_zero_cog_score}",
        filename = "nozerocogscore_{epoch:02d}-{val_loss:.4f}",
        monitor = "val_loss",
        mode = "min",
        save_top_k = 1,
        save_last = True
    )
    trainer_no_zero_cog_score = L.Trainer(
        logger=wandb_logger_no_zero_cog_score,
        max_epochs=n_epoch,
        accelerator=accelerator,
        log_every_n_steps=32,
        callbacks = [checkpoint_callback]
    )
    trainer_no_zero_cog_score.fit(model=lit_gnn_no_zero_cog_score, train_dataloaders=loader)
    wandb.finish()


if __name__ == "__main__":
    main()