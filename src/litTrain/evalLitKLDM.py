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
from src.dataLib.data_util import PyGData_to_Structure
from src.metrics.csp import CSPMetrics
from src.litTrain.trainLitKLDM import (
    build_transform,
    build_model,
    build_kldm,
    data_folder,
)
from src.lit.litCSPVNet import LitCSPVNet


# --------------------------------------------------------------------------- #
# Config (CLI-overridable defaults)
# --------------------------------------------------------------------------- #
DEFAULT_CKPT_GLOB = "checkpoints/*/CSPVNet_KLDM_*/*.ckpt"

# Reverse-diffusion sampling on CPU is robust and avoids missing MPS/CUDA TDM
# kernels; override with --device if you have full GPU kernel coverage.
DEFAULT_DEVICE = "cpu"

# Sampling hyper-parameters mirror the notebook generation cell.
SAMPLE_KWARGS = dict(
    fT_prior_kw="uniform",
    vT_prior_kw="stdGauss",
    lT_prior_kw="stdGauss",
    exponential_integration=True,
)

# StructureMatcher tolerances used for the CSP match rate (DiffCSP/CDVAE style).
MATCHER_KWARGS = dict(stol=0.5, angle_tol=10.0, ltol=0.3)


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


def kldm_output_to_structures(graph: Data, l0: torch.Tensor, f0: torch.Tensor):
    """Split the batched reverse-diffusion output into pymatgen Structures."""
    batch_vec = graph.batch.detach().cpu()
    h = graph.h.detach().cpu()
    f0 = f0.detach().cpu()
    l0 = l0.detach().cpu()

    structures = []
    for g in range(int(batch_vec.max().item()) + 1):
        mask = batch_vec == g
        data = Data(x=f0[mask], h=h[mask], l=l0[g].view(1, 6))
        structures.append(PyGData_to_Structure(data))
    return structures


@torch.inference_mode()
def evaluate(
    ckpt_path: str,
    split: str = "val",
    batch_size: int = 64,
    n_steps: int = 50,
    pc: bool = False,
    pc_steps: int = 10,
    max_batches: int | None = None,
    device_str: str = DEFAULT_DEVICE,
    seed: int | None = 0,
) -> dict:
    device = torch.device(device_str)

    transform = build_transform()
    ds_path = os.path.join(data_folder, f"{split}.pt")
    dataset = CrystalDataset(path=ds_path, transform=transform)
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    lit = load_lit_model(ckpt_path, device)

    def score_fn(graph, vt, t):
        """Joint score wrapper: (graph, vt, t) -> (score_l, score_f)."""
        return lit.forward_from_data(graph, vt, t)

    metrics = CSPMetrics(**MATCHER_KWARGS)

    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    pbar = tqdm(loader, total=n_batches, desc=f"CSP eval [{split}]")
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        batch = _cast_floats(batch.to(device))

        # Ground truth is read from the (unmodified) conditioning batch; the atom
        # types / counts are what we condition the generation on.
        target_structures = [PyGData_to_Structure(batch[j]) for j in range(batch.num_graphs)]
        if pc:
            l0, f0 = lit.kldm.sample_backward(
                graphT=batch,
                score_fn=score_fn,
                n_steps=n_steps,
                predictor_corrector=True,
                predictor_corrector_steps=pc_steps,
                seed=seed,
                **SAMPLE_KWARGS,
            )
        else:
    
            l0, f0 = lit.kldm.sample_backward(
                graphT=batch,
                score_fn=score_fn,
                n_steps=n_steps,
                seed=seed,
                **SAMPLE_KWARGS,
            )
        gen_structures = kldm_output_to_structures(batch, l0, f0)

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
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=1000, help="Reverse-diffusion steps.")
    parser.add_argument("--pc", action="store_true", help="Use predictor-corrector.")
    parser.add_argument("--pc-steps", type=int, default=20, help="Predictor-corrector steps.")
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
    print(f"Split: {args.split} | n_steps: {args.n_steps} | device: {args.device}")

    summary = evaluate(
        ckpt_path=ckpt_path,
        split=args.split,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
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
