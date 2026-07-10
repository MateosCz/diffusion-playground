"""
Evaluate the *crystal-structure-prediction* (CSP) quality of a trained
``LitCSPVNet`` / ``KLDM`` checkpoint.

For every crystal in the evaluation split we keep its atom types ``h`` (and atom
count) fixed as conditioning, then sample a lattice + fractional coordinates via
the reverse (denoising) ``KLDM`` process. Each generated structure is compared
against its ground-truth counterpart with :class:`src.metrics.csp.CSPMetrics`,
which reports:

  - ``valid``      : fraction of generated structures that are physically valid
  - ``match_rate`` : fraction that match the ground truth (StructureMatcher)
  - ``rmse``       : mean normalized RMS displacement over matched structures

The network / diffusion / transform are imported directly from
``src.litTrain.trainLitKLDM`` so the evaluation configuration can never drift
from the training configuration.
"""

import argparse
import glob
import os

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

from src.dataLib.realCrystal import CrystalDataset
from src.dataLib.data_util import PyGData_to_Structure, kldm_output_to_structures_batch
from src.metrics.csp import CSPMetrics
from src.litTrain.trainLitKLDM import (
    build_transform,
    build_model,
    build_kldm,
)
from src.lit.litCSPVNet import LitCSPVNet

dataset_name = "mp-20"
data_folder = "data/mp-20"
# --------------------------------------------------------------------------- #
# Config (CLI-overridable defaults)
# --------------------------------------------------------------------------- #
DEFAULT_CKPT_GLOB = f"checkpoints/*/CSPVNet_KLDM_{dataset_name}_*/*.ckpt"

# Reverse-diffusion sampling on CPU is robust and avoids missing MPS/CUDA TDM
# kernels; override with --device if you have full GPU kernel coverage.
DEFAULT_DEVICE = "mps"

# StructureMatcher tolerances used for the CSP match rate (DiffCSP/CDVAE style).
MATCHER_KWARGS = dict(stol=0.5, angle_tol=10.0, ltol=0.3)


def resolve_sample_kwargs(
    lit: LitCSPVNet,
    n_steps: int | None = None,
    pc: bool | None = None,
    pc_steps: int | None = None,
) -> dict:
    """Merge checkpoint ``sample_kwargs`` with optional CLI overrides.

    Training validation uses ``lit.sample()`` with the checkpoint kwargs. Eval
    must do the same; turning on predictor-corrector without a tuned step size
    makes the lattice state diverge (log-lengths >> 1 -> invalid / inf cells).
    """
    kwargs = dict(lit.sample_kwargs)
    if n_steps is not None:
        kwargs["n_steps"] = n_steps
    if pc is not None:
        kwargs["predictor_corrector"] = pc
        kwargs["predictor_corrector_n_steps"] = pc_steps if pc else 0
    elif pc_steps is not None:
        kwargs["predictor_corrector_n_steps"] = pc_steps
    return kwargs


def find_latest_checkpoint(pattern: str = DEFAULT_CKPT_GLOB) -> str:
    """Pick the most recent checkpoint (preferring ``last.ckpt``)."""
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched '{pattern}'.")
    last = [c for c in ckpts if c.endswith("last.ckpt")]
    return (last or ckpts)[-1]


def load_lit_model(ckpt_path: str, device: torch.device) -> LitCSPVNet:
    """Rebuild the exact training architecture and restore trained weights."""
    model = build_model()
    kldm = build_kldm().to(device)
    lit = LitCSPVNet.load_from_checkpoint(
        ckpt_path,
        model=model,
        kldm=kldm,
        map_location=device,
    )
    lit.eval()
    lit.to(device)
    return lit


def _cast_floats(graph: Data) -> Data:
    """Match float32 network weights (carbon-24 is stored in float64)."""
    for key, val in list(graph.items()):
        if torch.is_tensor(val) and torch.is_floating_point(val):
            graph[key] = val.float()
    return graph


@torch.inference_mode()
def evaluate(
    dataset_name: str,
    data_folder: str,
    ckpt_path: str,
    split: str = "val",
    batch_size: int = 64,
    n_steps: int | None = None,
    pc: bool | None = False,
    pc_steps: int | None = 1,
    max_batches: int | None = None,
    device_str: str = DEFAULT_DEVICE,
    seed: int | None = 0,
) -> dict:
    device = torch.device(device_str)

    transform = build_transform()

    transform_lengths, transform_angles, transform_pos = transform.transforms[0], transform.transforms[1], transform.transforms[2]
    ds_path = os.path.join(data_folder, f"{split}.pt")
    dataset = CrystalDataset(path=ds_path, transform=transform)
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    lit = load_lit_model(ckpt_path, device)
    sample_kwargs = resolve_sample_kwargs(lit, n_steps=n_steps, pc=pc, pc_steps=pc_steps)
    print(f"Sampling kwargs: {sample_kwargs}")

    metrics = CSPMetrics(**MATCHER_KWARGS)

    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    pbar = tqdm(loader, total=n_batches, desc=f"CSP eval [{split}]")
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        batch = _cast_floats(batch.to(device))

        # Ground truth is read from the (unmodified) conditioning batch; the atom
        # types / counts are what we condition the generation on.
        target_structures = [PyGData_to_Structure(batch[j], transform_lengths, transform_angles, transform_pos) for j in range(batch.num_graphs)]
        l0, f0 = lit.kldm.sample_backward(
            graphT=batch,
            score_fn=lit.forward_from_data,
            seed=seed,
            **sample_kwargs,
        )

        gen_structures = kldm_output_to_structures_batch(batch, l0, f0, transform_lengths, transform_angles, transform_pos)

        metrics.update(gen_structures, target_structures)
        summary = metrics.summarize()
        pbar.set_postfix(
            valid=f"{summary['valid']:.3f}",
            match=f"{summary['match_rate']:.3f}",
            rmse=f"{summary['rmse']:.4f}",
        )

    return metrics.summarize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a .ckpt file. Defaults to the latest CSPVNet_KLDM checkpoint.",
    )
    parser.add_argument("--dataset-name", type=str, default=dataset_name)
    parser.add_argument("--data-folder", type=str, default=data_folder)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Reverse-diffusion steps (default: value stored in the checkpoint).",
    )
    parser.add_argument(
        "--pc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable predictor-corrector (default: checkpoint setting).",
    )
    parser.add_argument(
        "--pc-steps",
        type=int,
        default=1,
        help="Predictor-corrector steps when --pc is enabled.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap the number of evaluated batches (quick smoke test).",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint()
    print(f"Evaluating checkpoint: {ckpt_path}")
    print(f"Dataset: {args.data_folder} | Split: {args.split} | device: {args.device}")

    summary = evaluate(
        dataset_name=args.dataset_name,
        data_folder=args.data_folder,
        ckpt_path=ckpt_path,
        split=args.split,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        pc=args.pc,
        pc_steps=args.pc_steps,
        max_batches=args.max_batches,
        device_str=args.device,
        seed=args.seed,
    )

    print("\n================ CSP evaluation ================")
    print(f"  validity   : {summary['valid']:.4f}")
    print(f"  match rate : {summary['match_rate']:.4f}")
    print(f"  rmse       : {summary['rmse']:.4f}")
    print("===============================================")


if __name__ == "__main__":
    main()
