"""Checkpoint helpers shared by the Lightning training entry points."""

from os import PathLike

from lightning.pytorch.callbacks import ModelCheckpoint


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
