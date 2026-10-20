"""Train an ambient coordinate/lattice flow map on MP20, conditioning on atom types."""

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
from src.dataLib.data_util import PyGData_to_Structure, kldm_output_to_structures_batch
from src.dataLib.realCrystal import CrystalDataset
from src.nn.ACSPNet import Ambient_CSPNet
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


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
total_time = 1.0
dim = 3
n_epoch = 3000
lr = 1e-3
batch_size = 64
num_workers = 1
eta = 0.5
evaluate_batch_size = 64
val_portion = 0.2
generation_eval_every_n_epochs = 10

data_folder = "data/mp-20"
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
    "time_dim": 256,
    "num_layers": 6,
    "h_dim": 512,       # embedding table size for atom types (indexed by atomic number)
    # "num_freqs": 128,
    "dis_emb_dim": 128,
    "ln": True,
    "smooth": False,    # embed discrete atom types via nn.Embedding
    "pred_h": False,    # <-- do not predict atom types
    "pred_l": True,
    "f_dim": 2 * dim,
    "radii_emb_dim": 128,
}

lit_kwargs = {
    "lambda_l": 1.0,
    "lambda_f": 1.5,
    "beta_l": 3.0,
    "beta_f": 3.0,

}

train_kwargs = {
    "with_ema": True,
    "ema_decay": 0.999,
    "ema_start": 500,
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


def build_model() -> Ambient_CSPNet:
    return Ambient_CSPNet(**nn_kwargs)


def build_flowmap() -> JointFlowMapCSP:
    return JointFlowMapCSP(
        f_flowmap=VFMFlowMap(RGVFM(FlatTorus01(dim=dim), **f_vfm_kwargs)),
        l_flowmap=VFMFlowMap(VFM(dim=6, **l_vfm_kwargs)),
        # # Self-distillation treats the diagonal endpoint prediction as a teacher.
        # stop_gradient_f=True,
    )



class FractionalPositions:
    """Position adapter for structure helpers: coordinates are already fractional."""

    @staticmethod
    def invert_one(positions):
        return positions % 1.0


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
        generated = pl_module.sample(batch, n_steps=self.n_steps)
        # Decode R^6 ambient pairs directly to fractional coordinates in [0, 1).
        fractional = pl_module.rg_vfm.to_intrinsic(generated.x)
        structures = kldm_output_to_structures_batch(
            batch, generated.l, fractional, *self.transforms,
        )
        self.metrics.update(structures, targets)


        # 每次验证只在第一个 batch 绘图；多卡时只由 rank 0 上传。
        if batch_idx != 0 or dataloader_idx != 0 or not trainer.is_global_zero:
            return

        logger = next(
            (lg for lg in trainer.loggers if isinstance(lg, WandbLogger)),
            None,
        )
        if logger is None:
            return

        num_images = 25

        # 成对选取，保证生成图和真实图的 Sample 编号对应。
        pairs = [
            (s, t) for s, t in zip(structures, targets) if s is not None
        ][:num_images]
        if not pairs:
            return

        for key, items in (
            ("val/images_generated", [s for s, _ in pairs]),
            ("val/images_gt", [t for _, t in pairs]),
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

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            pl_module.log_dict({f"val/{key}": value for key, value in self.metrics.summarize().items()})


def main():
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
        **train_kwargs,
    )

    experiment_name = f"ACSPVNet_FlowMap_{dataset_name}_eta_{eta}_lambda_l_{lit_kwargs['lambda_l']}_lambda_f_{lit_kwargs['lambda_f']}_beta_l_{lit_kwargs['beta_l']}_beta_f_{lit_kwargs['beta_f']}_zeroinitNN"
    wandb_logger = WandbLogger(
        name=experiment_name,
        save_dir="wandb_logs",
        project="diffusion-playground",
        checkpoint_name=experiment_name,
    )

    checkpoint_dir = f"checkpoints/{timestamp}/{experiment_name}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="flowmap_csp_{epoch:04d}-{val/rmse:.6f}",
        monitor="val/rmse",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

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
        ],
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
