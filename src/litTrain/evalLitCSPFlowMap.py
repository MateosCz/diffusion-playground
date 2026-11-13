"""
Evaluate the *crystal-structure-prediction* (CSP) quality of a trained
``LitCSPFlowMap`` checkpoint.

For every crystal in the evaluation split we keep its atom types ``h`` (and atom
count) fixed as conditioning, then sample a lattice + fractional coordinates via
the joint CSP flow map. Each generated structure is compared
against its ground-truth counterpart with :class:`src.metrics.csp.CSPMetrics`,
which reports:

  - ``valid``      : fraction of generated structures that are physically valid
  - ``match_rate`` : fraction that match the ground truth (StructureMatcher)
  - ``rmse``       : mean normalized RMS displacement over matched structures

Current network / flow map / transform defaults are imported from
``src.litTrain.trainLitCSPFlowMap``. Legacy learned-embedding networks are
reconstructed from their saved tensor shapes.
"""

import argparse
import glob
import os
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

from src.dataLib.realCrystal import CrystalDataset
from src.dataLib.data_util import PyGData_to_Structure, kldm_output_to_structures_batch
from src.metrics.csp import CSPMetrics
from src.litTrain.trainLitCSPFlowMap import (
    build_transform,
    build_model,
    build_flowmap,
    FractionalPositions,
    data_folder,
    dataset_name,
    lit_kwargs as default_lit_kwargs,
    sample_kwargs as default_sample_kwargs,
)
from src.lit.litCSPFlowMap import LitCSPFlowMap
from src.nn.ACSPNet import Ambient_CSPNet

from pymatgen.analysis.structure_matcher import StructureMatcher

# --------------------------------------------------------------------------- #
# Config (CLI-overridable defaults)
# --------------------------------------------------------------------------- #
DEFAULT_CKPT_GLOB = f"checkpoints/*/ACSPVNet_FlowMap_{dataset_name}_*/*.ckpt"

# Override with --device to use CPU or CUDA.
# DEFAULT_DEVICE = "cuda"
DEFAULT_DEVICE = "mps"

# StructureMatcher tolerances used for the CSP match rate (DiffCSP/CDVAE style).
MATCHER_KWARGS = dict(stol=0.5, angle_tol=10.0, ltol=0.3)

# Same tolerances as the reported metric, so best-of-N selection is consistent
# with how the winner will later be scored.
_SELECTION_MATCHER = StructureMatcher(**MATCHER_KWARGS)


def _candidate_rms(gen, target) -> float | None:
    """Normalized RMS displacement, or None if invalid / no match."""
    if gen is None:
        return None
    try:
        res = _SELECTION_MATCHER.get_rms_dist(gen, target)
    except Exception:
        return None
    return None if res is None else float(res[0])


def select_best_candidates(candidates, targets):
    """Pick, per crystal, the candidate with the lowest RMS against the target.

    ``candidates`` is a list of length ``best_n``; each element is a list of
    ``B`` generated structures (one per crystal in the batch). Crystals with no
    matching candidate fall back to the first sample, so they still contribute
    to ``valid`` / ``match_rate`` as a miss rather than being dropped.
    """
    best = []
    for j, target in enumerate(targets):
        best_k, best_rms = 0, None
        for k, cand in enumerate(candidates):
            rms = _candidate_rms(cand[j], target)
            if rms is None:
                continue
            if best_rms is None or rms < best_rms:
                best_rms, best_k = rms, k
        best.append(candidates[best_k][j])
    return best

def resolve_sample_kwargs(n_steps: int | None = None) -> dict:
    """Use training sampling defaults unless overridden on the command line."""
    kwargs = dict(default_sample_kwargs)
    if n_steps is not None:
        kwargs["n_steps"] = n_steps
    return kwargs


def find_latest_checkpoint(pattern: str = DEFAULT_CKPT_GLOB) -> str:
    """Pick the most recent checkpoint (preferring ``last.ckpt``)."""
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched '{pattern}'.")
    last = [c for c in ckpts if c.endswith("last.ckpt")]
    return (last or ckpts)[-1]


def _build_legacy_model(state_dict: dict) -> Ambient_CSPNet:
    """Rebuild the original learned-embedding network from its saved shapes."""
    weights = {key.removeprefix("pi."): value for key, value in state_dict.items() if key.startswith("pi.")}
    embedding = weights["node_embedding.weight"]
    layer_indices = {int(key.split(".")[1]) for key in weights if key.startswith("layers.")}
    return Ambient_CSPNet(
        hidden_dim=embedding.shape[1],
        h_dim=embedding.shape[0] - 1,
        time_dim=2 * weights["time_emb.weight"].shape[1],
        num_layers=max(layer_indices) + 1 if layer_indices else 0,
        dis_emb_dim=weights["dis_emb.0.weight"].shape[0],
        radii_emb_dim=weights["radii_emb.0.weight"].shape[0],
        f_dim=weights["out_f.2.weight"].shape[0],
        ln="final_layer_norm.weight" in weights,
        smooth=False,
        pred_h="out_h.weight" in weights,
        pred_l="out_l.weight" in weights,
    )


def load_lit_model(
    ckpt_path: str, device: torch.device, type_encoding_state_path: str | None = None,
) -> LitCSPFlowMap:
    """Rebuild the exact training architecture and restore trained weights."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = checkpoint["state_dict"]
    legacy_embedding = (
        "pi.node_embedding.weight" in weights
        and "pi.atom_type_emb_mlp.0.weight" not in weights
    )
    if legacy_embedding:
        if type_encoding_state_path is not None:
            raise ValueError("This learned-embedding checkpoint does not use a PCA encoding file.")
        model = _build_legacy_model(weights)
    else:
        encoding_state = (
            torch.load(type_encoding_state_path, map_location="cpu", weights_only=True)
            if type_encoding_state_path is not None else None
        )
        model = build_model(type_encoding_state=encoding_state, initialize_type_encoding=False)
    del weights, checkpoint
    flow_map = build_flowmap()
    lit = LitCSPFlowMap.load_from_checkpoint(
        ckpt_path,
        pi=model,
        flow_map=flow_map,
        use_weight_mlp=default_lit_kwargs["use_weight_mlp"],
        map_location=device,
        weights_only=False
    )
    lit.eval()
    lit.to(device)
    return lit


@torch.inference_mode()
def evaluate(
    dataset_name: str,
    data_folder: str,
    ckpt_path: str,
    split: str = "val",
    batch_size: int = 64,
    portion: float = 1.0,
    n_steps: int | None = None,
    max_batches: int | None = None,
    best_n: int | None = 1,
    device_str: str = DEFAULT_DEVICE,
    seed: int | None = 0,
    type_encoding_state_path: str | None = None,
) -> dict:
    device = torch.device(device_str)

    transform = build_transform(data_folder_override=data_folder)

    transform_lengths, transform_angles = transform.transforms[:2]
    transform_pos = FractionalPositions()
    ds_path = os.path.join(data_folder, f"{split}.pt")
    dataset = CrystalDataset(path=ds_path, transform=transform, portion=portion)
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    lit = load_lit_model(ckpt_path, device, type_encoding_state_path=type_encoding_state_path)
    encoding_state = (
        lit.pi.get_type_encoding_state() if hasattr(lit.pi, "get_type_encoding_state") else None
    )
    if encoding_state is not None:
        print(f"Atom encoding: {lit.pi.atom_type_rep} | state_sha256={encoding_state['state_sha256']}")
    elif isinstance(lit.pi, Ambient_CSPNet):
        print("Atom encoding: learned embedding (restored from checkpoint)")
    sample_kwargs = resolve_sample_kwargs(n_steps=n_steps)
    print(f"Sampling kwargs: {sample_kwargs}")

    if seed is not None:
        torch.manual_seed(seed)

    metrics = CSPMetrics(**MATCHER_KWARGS)

    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    pbar = tqdm(loader, total=n_batches, desc=f"CSP eval [{split}]")
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        batch = batch.to(device)

        # Ground truth is read from the (unmodified) conditioning batch; the atom
        # types / counts are what we condition the generation on.
        target_structures = [PyGData_to_Structure(batch[j], transform_lengths, transform_angles, transform_pos) for j in range(batch.num_graphs)]

        gen_struct_candidates = []
        for _ in range(best_n):
            generated = lit.sample(batch, **sample_kwargs)
            fractional = lit.rg_vfm.to_intrinsic(generated.x)

            gen_struct_candidates.append(kldm_output_to_structures_batch(batch, generated.l, fractional, transform_lengths, transform_angles, transform_pos))
        if best_n > 1:
            gen_structures = select_best_candidates(gen_struct_candidates, target_structures)
        else:
            gen_structures = gen_struct_candidates[0]
        metrics.update(gen_structures, target_structures)

        summary = metrics.summarize()
        pbar.set_postfix(
            valid=f"{summary['valid']:.3f}",
            match=f"{summary['match_rate']:.3f}",
            rmse=f"{summary['rmse']:.4f}" if summary["rmse"] is not None else "None",
        )

    return metrics.summarize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a .ckpt file. Defaults to the latest ACSPVNet_FlowMap checkpoint.",
    )
    parser.add_argument("--dataset-name", type=str, default=dataset_name)
    parser.add_argument("--data-folder", type=str, default=data_folder)
    parser.add_argument(
        "--type-encoding-state", type=str, default=None,
        help="Original training encoding file, required only for legacy subatomic-tokenizer checkpoints without saved encoding state.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Flow-map steps (default: training sampling setting, currently 1).",
    )
    parser.add_argument(
        "--portion",
        type=float,
        default=1.0,
        help="Portion of the dataset to evaluate (default: 1.0).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap the number of evaluated batches (quick smoke test).",
    )
    parser.add_argument(
        "--best-n",
        type=int,
        default=1,
        help="Sample N structures per crystal and keep the best-matching one.",
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
        portion=args.portion,
        n_steps=args.n_steps,
        best_n=args.best_n,
        max_batches=args.max_batches,
        device_str=args.device,
        seed=args.seed,
        type_encoding_state_path=args.type_encoding_state,
    )

    print("\n================ CSP evaluation ================")
    print(f"  validity   : {summary['valid']:.4f}")
    print(f"  match rate : {summary['match_rate']:.4f}")
    rmse_text = f"{summary['rmse']:.4f}" if summary["rmse"] is not None else "None (no matched structures)"
    print(f"  rmse       : {rmse_text}")
    print("===============================================")


if __name__ == "__main__":
    main()
