"""Train an ambient coordinate/lattice flow map on MP20, conditioning on atom types."""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch_geometric.transforms as PyGT
from torch_geometric.loader import DataLoader as PyGDataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import src.dataLib.transform as T
from src.dataLib.data_util import PyGData_to_Structure, kldm_output_to_structures_batch
from src.dataLib.realCrystal import CrystalDataset
from src.nn.ACSPNet import Ambient_CSPNet
from src.nn.ACSPNet_Token import Ambient_CSPNet_Token
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching.vfm import VFM
from src.flow_matching.rg_vfm import RGVFM
from src.csp_model.flowmap_csp import JointFlowMapCSP
from src.lit.litCSPFlowMap import LitCSPFlowMap
from src.lit.callbacks import last_checkpoint
from src.manifolds import FlatTorus01
from src.metrics.csp import CSPMetrics
from src.device import get_default_device, get_lightning_accelerator
from matplotlib import pyplot as plt
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
total_time = 1.0
dim = 3
n_epoch = 8000
lr = 5e-4
batch_size = 128 * 4
num_workers = 1
eta = 0.75
evaluate_batch_size = 64
val_portion = 0.2 
generation_eval_every_n_epochs = 25
# Load model/EMA/WeightMLP weights into a fresh training run.
init_ckpt = None
# Use only for a full resume from a compatible new-format training checkpoint.
continue_ckpt = None
# continue_ckpt = "checkpoints/20261102_172152_cspFlowMap/ACSPVNet_FlowMap_mp-20_learnable_edm2_linear_weight_lambda_l_0.05_lambda_f_1.0_lr_0.001_batch_size_512/flowmap_csp_2879-0.126604.ckpt"
# continue_ckpt = "checkpoints/20261102_181931_cspFlowMap/ACSPVNet_FlowMap_mp-20_learnable_edm2_linear_weight_lambda_l_0.005_lambda_f_1.0_lr_0.005_batch_size_512/flowmap_csp_2929-0.126363.ckpt"
data_folder = "data/mp-20/tokenized"
dataset_name = "mp-20"

l_vfm_kwargs = {
    "time_eps": 1e-3,
    "is_loss_weighted": True,
}
f_vfm_kwargs = {
    "total_time": total_time,

    "time_eps": 1e-3,
    "max_velocity_scale": None,
    "max_loss_weight": 100.0,
    "support": "extrinsic",
    "ambient_metric": "euclidean",
}


sample_kwargs = {
    "n_steps": 4,
}

nn_kwargs = {
    "hidden_dim": 512,
    # "hidden_dim": 256,
    # "hidden_dim": 256,
    "time_dim": 256,
    "num_layers": 6,
    # "num_layers": 4,
    # "num_layers": 4,
    "h_dim": 512,       # embedding table size for atom types (indexed by atomic number) (should be the same as the number of atom types)
    # "num_freqs": 128,
    "dis_emb_dim": 128,
    "ln": True,
    # "smooth": False,    # embed discrete atom types via nn.Embedding
    "pred_h": False,    # <-- do not predict atom types
    "pred_l": True,
    "f_dim": 2 * dim,
    "radii_emb_dim": 128,
    # "neighbor_head": True,
    # "neighbor_hidden_dim": 128,
    "atom_type_rep": "subatomic_tokenizer_pca_16",
    "weight_mlp_kwargs": {
        "sin_emb_dim": 256,
    },
    # "composition_context": True,
    # "context_hidden_dim": 128,
}

lit_kwargs = {
    "use_weight_mlp": True,
    "lambda_l": 0.05,
    "lambda_f": 1.0,
    "beta_l": 0.1,
    "beta_f": 0.05,
    "non_linear_weight_mlp": False,

}

train_kwargs = {
    "with_ema": True,
    "ema_decay": 0.999,
    "ema_start": 500 if continue_ckpt is None else -1,
    "warmup_steps": 1000,

}


def build_transform() -> PyGT.Compose:
    return PyGT.Compose(
        [
            T.ContinuousIntervalLengths(in_key="lengths", out_key="lengths", lengths_loc_scale= f"{data_folder}/train_mean_std_stats.json"),
            T.ContinuousIntervalAngles(in_key="angles", out_key="angles",angles_loc_scale= (0.0, 0.35) ),
            # Keep fractional positions; RGVFM embeds them as cos/sin(2*pi*f).
            T.FullyConnectedGraph(key="edge_index", len_from="pos"),
            # lattice vector l = (log a, log b, log c, tan alpha, tan beta, tan gamma)
            T.ConcatFeatures(in_keys=["lengths", "angles"], out_key="l"),
            # crystal data is stored in float64; match float32 network weights
            T.CastFloating(dtype=torch.float32),
        ]
    )


def build_model() -> Ambient_CSPNet | Ambient_CSPNet_Token:
    # return Ambient_CSPNet(**{
    #     key: value for key, value in nn_kwargs.items() if key != "weight_mlp_kwargs"
    # })
    return Ambient_CSPNet_Token(**{
        key: value for key, value in nn_kwargs.items() if key != "weight_mlp_kwargs"
    })


def build_flowmap() -> JointFlowMapCSP:
    return JointFlowMapCSP(
        f_flowmap=VFMFlowMap(RGVFM(FlatTorus01(dim=dim), **f_vfm_kwargs)),
        l_flowmap=VFMFlowMap(VFM(dim=6, **l_vfm_kwargs)),
        # # Self-distillation treats the diagonal endpoint prediction as a teacher.
        # stop_gradient_f=True,
    )


def load_initial_weights(lit_model, checkpoint_path):
    """Load and migrate weights without restoring optimizer, scheduler, or loops."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lit_model.on_load_checkpoint(checkpoint)
    lit_model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(
        f"Initialized weights from {checkpoint_path} "
        f"(source epoch={checkpoint.get('epoch')}, "
        f"global_step={checkpoint.get('global_step')}); "
        "optimizer, scheduler, and training epoch start fresh."
    )

class FractionalPositions:
    """Position adapter for structure helpers: coordinates are already fractional."""

    @staticmethod
    def invert_one(positions):
        return positions % 1.0


class LossSpikeCheckpoint(L.Callback):
    """Keep the largest single-device loss and its sampled inputs before backward."""

    def __init__(self, threshold=10.0):
        super().__init__()
        self.max_loss = float(threshold)

    def on_before_backward(self, trainer, pl_module, loss):
        # save_checkpoint can be collective; this diagnostic supports one device.
        if trainer.world_size != 1:
            return
        value = float(loss.detach().cpu())
        if not value > self.max_loss:
            return
        directory = Path(trainer.checkpoint_callback.dirpath)
        directory.mkdir(parents=True, exist_ok=True)
        inputs = {
            key: tuple(item.clone().detach().cpu() for item in items)
            for key, items in pl_module._loss_spike_inputs.items()
        }
        self.max_loss = value
        trainer.save_checkpoint(str(directory / "loss_spike.ckpt"))
        torch.save(
            {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "loss": value,
                "inputs": inputs,
            },
            directory / "loss_spike_inputs.pt",
        )

class WeightMLPHeatmap(L.Callback):
    """Log w(s,t) and exp(-w(s,t)) for both loss-weight networks."""

    def __init__(self, n_grid=101):
        super().__init__()
        if n_grid < 2:
            raise ValueError("n_grid must be >= 2")
        self.n_grid = n_grid

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        # skip sanity check and only plot and upload by rank 0 when multi-card training
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if not pl_module.use_weight_mlp:
            return

        logger = next(
            (lg for lg in trainer.loggers if isinstance(lg, WandbLogger)),
            None,
        )
        if logger is None:
            return

        t_max = float(pl_module.rg_vfm.total_time - pl_module.rg_vfm.time_eps)

        for branch in ("f", "l"):
            net = getattr(pl_module, f"weight_mlp_{branch}")
            ref = next(net.parameters())

            times = torch.linspace(
                0.0, t_max, self.n_grid,
                device=ref.device, dtype=ref.dtype,
            )

            # the matrix rows correspond to s, columns correspond to t; only calculate the s <= t region used for training
            s_grid, t_grid = torch.meshgrid(times, times, indexing="ij")
            valid = s_grid <= t_grid

            s = s_grid[valid].reshape(-1, 1)
            t = t_grid[valid].reshape(-1, 1)

            w = net(s, t).reshape(-1).float()
            mask = valid.cpu().numpy()

            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5), constrained_layout=True,
            )
            try:
                plots = (
                    (w, "MLP output: w(s,t)", "coolwarm"),
                    (
                        torch.exp(-w),
                        "Loss coefficient: exp(-w(s,t))",
                        "viridis",
                    ),
                )

                for ax, (values, title, cmap_name) in zip(axes, plots):
                    values = values.detach().cpu().numpy()

                    matrix = np.full((self.n_grid, self.n_grid), np.nan)
                    matrix[mask] = values

                    cmap = plt.get_cmap(cmap_name).copy()
                    cmap.set_bad("#eeeeee")

                    im = ax.imshow(
                        np.ma.masked_invalid(matrix),
                        origin="lower",
                        extent=(0, t_max, 0, t_max),
                        interpolation="nearest",
                        aspect="equal",
                        cmap=cmap,
                    )
                    ax.plot([0, t_max], [0, t_max], "k--", lw=0.8)
                    ax.set_xlabel("t")
                    ax.set_ylabel("s")

                    # if the network output contains NaN/Inf, display the number in the title
                    n_bad = int((~np.isfinite(values)).sum())
                    ax.set_title(
                        title + (f" | non-finite={n_bad}" if n_bad else "")
                    )
                    fig.colorbar(im, ax=ax)

                label = "coordinates" if branch == "f" else "lattice"
                fig.suptitle(
                    f"WeightMLP {branch} ({label})"
                    f" | epoch {trainer.current_epoch + 1}"
                )

                logger.log_image(
                    key=f"val/weight_mlp_{branch}",
                    images=[fig],
                    step=trainer.global_step,
                )
            finally:
                plt.close(fig)
class CSPGenerationMetrics(L.Callback):
    """Evaluate sampled crystals with the same structure metrics as KLDM."""

    def __init__(self, transform, n_steps=1):
        super().__init__()
        self.transforms = (*transform.transforms[:2], FractionalPositions())
        self.n_steps = n_steps
        self.metrics = CSPMetrics()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.metrics.reset()

    @torch.inference_mode()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ):
        if trainer.sanity_checking:
            return
        targets = [
            PyGData_to_Structure(graph, *self.transforms)
            for graph in batch.to_data_list()
        ]
        generated = pl_module.sample(batch, n_steps=self.n_steps, pc=False)
        # Decode R^6 ambient pairs directly to fractional coordinates in [0, 1).
        fractional = pl_module.rg_vfm.to_intrinsic(generated.x)
        structures = kldm_output_to_structures_batch(
            batch, generated.l, fractional, *self.transforms,
        )
        self.metrics.update(structures, targets)


        # only plot the first batch; only rank 0 uploads when multi-card
        if batch_idx != 0 or dataloader_idx != 0 or not trainer.is_global_zero:
            return

        logger = next(
            (lg for lg in trainer.loggers if isinstance(lg, WandbLogger)),
            None,
        )
        if logger is None:
            return
        if (pl_module.current_epoch + 1) % 50 != 0:
            return

        num_images = 25

        # sample pairs of generated and target structures, make sure the orders are same
        pairs = [
            (s, t) for s, t, valid in zip(
                structures, targets, self.metrics.valid[-len(structures):],
            ) if s is not None and valid
        ]
        if not pairs:
            return
        plot_matcher = StructureMatcher(
            primitive_cell=False,
            scale=False,
            ltol=0.3,
            stol=0.5,
            angle_tol=10,
        )

        aligned_pairs = []
        for generated, gt in pairs:
            try:
                aligned = plot_matcher.get_s2_like_s1(gt, generated)
            except (ZeroDivisionError, np.linalg.LinAlgError, ValueError) as error:
                warnings.warn(
                    f"Skipping plot alignment at epoch {pl_module.current_epoch}, "
                    f"batch {batch_idx}: {type(error).__name__}: {error}; "
                    f"volume={generated.volume}, lattice={generated.lattice.parameters}",
                    RuntimeWarning,
                )
                continue

            if aligned is not None:

                aligned = Structure(
                    lattice=Lattice.from_parameters(*aligned.lattice.parameters),
                    species=aligned.species,
                    coords=aligned.frac_coords,
                    coords_are_cartesian=False,
                    to_unit_cell=False,
                )

            aligned_pairs.append(
                (aligned if aligned is not None else generated, gt)
            )

        if not aligned_pairs:
            return
        pairs = aligned_pairs
        plot_pairs = pairs[:num_images]

        for key, items in (
            ("val/images_generated", [s for s, _ in plot_pairs]),
            ("val/images_gt", [t for _, t in plot_pairs]),
        ):
            ncols = min(5, len(items))
            nrows = (len(items) + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(3 * ncols, 3 * nrows),
                squeeze=False,
            )
            try:
                for ax in axes.flat:
                    ax.set_axis_off()

                for i, (ax, structure) in enumerate(
                    zip(axes.flat, items), start=1,
                ):
                    atoms = AseAtomsAdaptor.get_atoms(structure)
                    atoms.wrap()
                    plot_atoms(atoms, ax=ax)
                    ax.set_title(f"Sample {i}")

                fig.tight_layout()
                logger.log_image(
                    key,
                    [fig],
                    step=trainer.global_step,
                )
            finally:
                plt.close(fig)

        # Reuse the plotted pairs, including the fallback when alignment fails.
        pl_module.log_dict(
            {
                "val/lattice_angle_mae": float(np.mean([
                    np.abs(np.asarray(s.lattice.angles) - np.asarray(t.lattice.angles))
                    for s, t in aligned_pairs
                ])),  # degrees
                "val/lattice_length_mae": float(np.mean([
                    np.abs(np.asarray(s.lattice.abc) - np.asarray(t.lattice.abc))
                    for s, t in aligned_pairs
                ])),  # Angstrom
            },
            on_step=False, on_epoch=True, batch_size=len(aligned_pairs), rank_zero_only=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            # Lightning cannot log None; NaN represents the undefined displacement.
            pl_module.log_dict({
                f"val/{key}": float("nan") if value is None else value
                for key, value in self.metrics.summarize().items()
            })

class SaveTrainingConfig(L.Callback):
    def __init__(self, config, checkpoint_dir):
        super().__init__()
        self.config = config
        self.checkpoint_dir = checkpoint_dir

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.checkpoint_dir, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)


def main():
    if init_ckpt is not None and continue_ckpt is not None:
        raise ValueError("Set only one of init_ckpt and continue_ckpt.")
    effective_train_kwargs = dict(train_kwargs)
    if init_ckpt is not None:
        # Retained EMA must track the model immediately when the epoch restarts at 0.
        effective_train_kwargs["ema_start"] = -1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = get_default_device()
    accelerator = get_lightning_accelerator(device)

    transform = build_transform()
    train_ds = CrystalDataset(path=os.path.join(data_folder, "train.pt"), transform=transform)
    val_ds = CrystalDataset(path=os.path.join(data_folder, "val.pt"), transform=transform, portion=val_portion)

    train_loader = PyGDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # The two flow-map losses need at least two crystals.
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    val_loader = PyGDataLoader(
        val_ds,
        batch_size=evaluate_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    lit_model = LitCSPFlowMap(
        pi=build_model(),
        flow_map=build_flowmap(),
        eta=eta,
        lr=lr,
        lambda_l=lit_kwargs["lambda_l"],
        lambda_f=lit_kwargs["lambda_f"],
        beta_l=lit_kwargs["beta_l"],
        beta_f=lit_kwargs["beta_f"],
        use_weight_mlp=lit_kwargs["use_weight_mlp"],
        non_linear_weight_mlp=lit_kwargs["non_linear_weight_mlp"],
        weight_mlp_kwargs=nn_kwargs.get("weight_mlp_kwargs"),
        **effective_train_kwargs,
    )
    if init_ckpt is not None:
        load_initial_weights(lit_model, init_ckpt)

    experiment_name = f"ACSPNet_{dataset_name}_learnable_weight_time_diff_balanced_dim_linear_lmlp_no_sin_posdiff_lambda_l_{lit_kwargs['lambda_l']}_lambda_f_{lit_kwargs['lambda_f']}_lr_{lr}_batch_size_{batch_size}_subatomic_tokenizer_pca_16"
    if lit_kwargs["non_linear_weight_mlp"]:
        experiment_name += "_non_linear_weight_mlp"
    wandb_logger = WandbLogger(
        name=experiment_name,
        save_dir="wandb_logs",
        project="diffusion-playground",
        checkpoint_name=experiment_name,
    )

    checkpoint_dir = f"checkpoints/{timestamp}_cspFlowMap/{experiment_name}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="flowmap_csp_{epoch:04d}-{val/rmse:.6f}",
        monitor="val/rmse",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    config = {
        "dataset_name": dataset_name,
        "data_folder": data_folder,
        "dim": dim,
        "total_time": total_time,
        "n_epoch": n_epoch,
        "lr": lr,
        "eta": eta,
        "batch_size": batch_size,
        "evaluate_batch_size": evaluate_batch_size,
        "val_portion": val_portion,
        "num_workers": num_workers,
        "generation_eval_every_n_epochs": generation_eval_every_n_epochs,
        "init_ckpt": init_ckpt,
        "continue_ckpt": continue_ckpt,
        "accelerator": accelerator,
        "nn_kwargs": nn_kwargs,
        "lit_kwargs": lit_kwargs,
        "train_kwargs": effective_train_kwargs,
        "sample_kwargs": sample_kwargs,
        "l_vfm_kwargs": l_vfm_kwargs,
        "f_vfm_kwargs": f_vfm_kwargs,
        "transforms": {
            "lengths_loc_scale": f"{data_folder}/train_mean_std_stats.json",
            "angles_loc_scale": transform.transforms[1].angles_loc_scale,
            "positions": "fractional",
        },
    }

    trainer = L.Trainer(
        logger=wandb_logger,
        inference_mode=False,  # Validation flow-map losses also differentiate time.
        max_epochs=n_epoch,
        accelerator=accelerator,
        log_every_n_steps=32,
        check_val_every_n_epoch=generation_eval_every_n_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=4,
        callbacks=[
            checkpoint_callback, last_checkpoint(checkpoint_dir),
            CSPGenerationMetrics(transform, **sample_kwargs),
            LossSpikeCheckpoint(),
            SaveTrainingConfig(config, checkpoint_dir),
            WeightMLPHeatmap(n_grid=100),
        ],
    )

    try:
        trainer.fit(
            model=lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=continue_ckpt,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
