"""Checkpoint helpers shared by the Lightning training entry points."""

from os import PathLike

from lightning.pytorch.callbacks import ModelCheckpoint
import os
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import ase
import numpy as np
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from src.dataLib.data_util import kldm_output_to_structures_batch, PyGData_to_Structure, save_images, save_json
import torch

def last_checkpoint(dirpath: str | PathLike[str]) -> ModelCheckpoint:
    """Return a callback that overwrites ``last.ckpt`` after every train epoch.

    Recent Lightning versions only update ``save_last=True`` when the callback
    also saves a top-k checkpoint.  A separate, unmonitored callback with a
    fixed filename keeps the latest completed epoch independently of whether a
    monitored metric improved.
    """
    return ModelCheckpoint(
        dirpath=dirpath,
        filename="last",
        monitor=None,
        save_top_k=1,
        save_last=False,
        save_on_exception=True,
        every_n_epochs=100,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )



def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    wandb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            if wandb_logger is not None:
                raise ValueError(
                    "More than one WandbLogger was found in the list of loggers"
                )

            wandb_logger = logger

    return wandb_logger


def make_atoms_grid(atoms_lst: list[ase.Atoms]):
    bs = len(atoms_lst)
    nrows = ncols = int(np.ceil(np.sqrt(bs)))

    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    for i, ax in enumerate(fig.axes):
        if i >= len(atoms_lst):
            break
        atoms = atoms_lst[i]
        plot_atoms(atoms, ax)
    return fig



# class LogSampledAtomsCallback(Callback):
#     def __init__(
#             self,
#             dirpath: Union[Path, str],
#             save_atoms: bool = True,
#             num_log_wandb: int = 25,
#             log_wandb_gt: bool = True,
#             prefix_with_epoch: bool = True,
#             n_steps: int = 4,
#             transforms = None,
#     ):
#         self.dirpath = dirpath
#         self.save_atoms = save_atoms
#         self.num_log_wandb = num_log_wandb
#         self.log_wandb_gt = log_wandb_gt
#         self.prefix_with_epoch = prefix_with_epoch
#         self.n_steps = n_steps
#         self.atoms_lst: list[ase.Atoms] = ...
#         self.atoms_lst_gt: list[ase.Atoms] = ...
#         self.transforms = transforms
#     def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         super().on_validation_start(trainer, pl_module)
#         self.atoms_lst = []
#         self.atoms_lst_gt = []

#     def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         return self.on_validation_start(trainer, pl_module)

#     def on_predict_epoch_start(
#             self, trainer: Trainer, pl_module: LightningModule
#     ) -> None:
#         return self.on_validation_start(trainer, pl_module)
        
#     @torch.inference_mode()
#     def on_validation_batch_end(
#             self,
#             trainer: Trainer,
#             pl_module: LightningModule,
#             outputs: list[ase.Atoms | Structure | None],
#             batch: Any,
#             batch_idx: int,
#             dataloader_idx: int = 0,
#     ) -> None:
#         if trainer.sanity_checking:
#             return
#         targets = [
#             PyGData_to_Structure(graph, *self.transforms)
#             for graph in batch.to_data_list()
#         ]
#         generated = pl_module.sample(batch, n_steps=self.n_steps)
#         fractional = pl_module.rg_vfm.to_intrinsic(generated.x)

#         structures = kldm_output_to_structures_batch(
#             batch, generated.l, fractional, *self.transforms,
#         )
#         if len(structures):
#             if isinstance(structures[0], Structure):
#                 adaptor = AseAtomsAdaptor()
#                 atoms = [adaptor.get_atoms(s) for s in structures]
#             else:
#                 atoms = structures

#             for a in atoms:
#                 try:
#                     a.wrap()
#                 except Exception as e:
#                     warnings.warn(
#                         f"In {self.__class__} the following error occurred: {e}"
#                     )

#             self.atoms_lst.extend(atoms)

#             if self.log_wandb_gt:
#                 adaptor = AseAtomsAdaptor()
#                 self.atoms_lst_gt.extend([adaptor.get_atoms(s) for s in structures_gt])

#     def on_test_batch_end(
#             self,
#             trainer: Trainer,
#             pl_module: LightningModule,
#             outputs: list[ase.Atoms | Structure | None],
#             batch: Any,
#             batch_idx: int,
#             dataloader_idx: int = 0,
#     ) -> None:
#         return self.on_validation_batch_end(
#             trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#         )

#     def on_predict_batch_end(
#             self,
#             trainer: Trainer,
#             pl_module: LightningModule,
#             outputs: list[ase.Atoms | Structure | None],
#             batch: Any,
#             batch_idx: int,
#             dataloader_idx: int = 0,
#     ) -> None:
#         return self.on_validation_batch_end(
#             trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#         )

#     def on_validation_epoch_end(
#             self, trainer: Trainer, pl_module: LightningModule
#     ) -> None:

#         super().on_validation_epoch_end(trainer, pl_module)
#         epoch = pl_module.current_epoch

#         if self.prefix_with_epoch:
#             dirpath = os.path.join(self.dirpath, str(epoch))
#         else:
#             dirpath = self.dirpath

#         if not os.path.exists(dirpath):
#             os.makedirs(dirpath, exist_ok=True)

#         if self.save_atoms:  # only save generated ones
#             save_path = os.path.join(dirpath, "samples.xyz")
#             save_images(self.atoms_lst, filename=save_path)

#         if self.num_log_wandb:
#             logger: WandbLogger = get_wandb_logger(trainer)

#             idx = min(len(self.atoms_lst), self.num_log_wandb)

#             fig = make_atoms_grid(self.atoms_lst[-idx:])
#             logger.log_image(f"val/images_generated", [fig])
#             plt.close(fig)

#             if self.log_wandb_gt and len(self.atoms_lst_gt):
#                 fig = make_atoms_grid(self.atoms_lst_gt[-idx:])
#                 logger.log_image(f"val/images_gt", [fig])
#                 plt.close(fig)

#     def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         return self.on_validation_epoch_end(trainer, pl_module)

#     def on_predict_epoch_end(
#             self, trainer: Trainer, pl_module: LightningModule, outputs: list[Any]
#     ) -> None:
#         return self.on_validation_epoch_end(trainer, pl_module)