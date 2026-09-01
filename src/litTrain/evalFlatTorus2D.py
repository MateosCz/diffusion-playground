"""Evaluate RG-VFM or RFM checkpoints on the 2D flat-torus checkerboard."""

import argparse
import json
import re
from pathlib import Path
from typing import Literal

import torch

from src.device import get_default_device
from src.flow_matching import RFM, RGVFM
from src.lit.checkerboard_generation_metrics import (
    checkerboard_distribution_metrics,
)
from src.manifolds import FlatTorus01
from src.nn.rfm_mlp import RFMMLP
from src.nn.rg_vfm_mlp import RGVFMMLP


Method = Literal["rfm", "rgvfm"]


def _checkpoint_metric(path: Path) -> float:
    match = re.search(r"-(\d+(?:\.\d+)?)\.ckpt$", path.name)
    return float(match.group(1)) if match else float("inf")


def find_checkpoint(project_root: Path, method: Method) -> Path:
    """Prefer the best distribution-selected checkpoint when available."""
    experiment = (
        "RFMMLP_checkerboard_fractional"
        if method == "rfm"
        else "RGVFMMLP_checkerboard_fractional"
    )
    roots = list((project_root / "checkpoints").glob(f"*/{experiment}"))
    distribution = [
        path
        for root in roots
        for path in root.glob("*distribution_*.ckpt")
    ]
    if distribution:
        return min(distribution, key=_checkpoint_metric)

    loss_checkpoints = [
        path
        for root in roots
        for path in root.glob("*.ckpt")
        if path.name != "last.ckpt"
    ]
    if loss_checkpoints:
        return min(loss_checkpoints, key=_checkpoint_metric)
    raise FileNotFoundError(f"no {method.upper()} checkerboard checkpoint found")


def _model_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }


def build_model_from_checkpoint(
    checkpoint: dict,
    *,
    method: Method,
    manifold: FlatTorus01,
) -> torch.nn.Module:
    """Infer architecture widths and periodic encoding from saved weights."""
    state = _model_state(checkpoint)
    input_features = state["lifting_layer_x.0.weight"].shape[1]
    x_lifting_dim = state["lifting_layer_x.0.weight"].shape[0]
    time_embedding_dim = state["lifting_layer_t.0.weight"].shape[1]
    hidden_dim = [state["lifting_layer_hidden.weight"].shape[0]]
    layer_ids = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith("endpoint_net.") and key.endswith(".weight")
    )
    hidden_dim.extend(
        state[f"endpoint_net.{layer_id}.weight"].shape[0]
        for layer_id in layer_ids
    )

    with_sincos_position = input_features != manifold.intrinsic_dim
    # The frequency buffer is present even in legacy raw-coordinate models,
    # so use it to preserve strict checkpoint compatibility.
    position_fourier_bands = state["position_frequencies"].numel()
    model_class = RFMMLP if method == "rfm" else RGVFMMLP
    model = model_class(
        dim=manifold.intrinsic_dim,
        x_lifting_dim=x_lifting_dim,
        time_embedding_half_dim=time_embedding_dim // 2,
        hidden_dim=hidden_dim,
        output_dim=state["output_layer.weight"].shape[0],
        total_time=1.0,
        time_embedding_scale=1.0,
        position_fourier_bands=position_fourier_bands,
        position_period=manifold.period,
        with_sincos_position=with_sincos_position,
        manifold=manifold,
    )
    model.load_state_dict(state, strict=True)
    return model


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    method: Method,
    n_samples: int = 4_096,
    n_steps: int = 100,
    seed: int = 2026,
) -> dict[str, float | int | str]:
    if n_samples < 1 or n_steps < 1:
        raise ValueError("n_samples and n_steps must be positive")

    device = get_default_device()
    manifold = FlatTorus01(dim=2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model_from_checkpoint(
        checkpoint,
        method=method,
        manifold=manifold,
    ).to(device)
    model.eval()

    if method == "rfm":
        flow = RFM(
            manifold,
            normalize_loss=False,
            integrator="euler",
        )
    else:
        flow = RGVFM(
            manifold,
            noise_scale=0.001,
            max_velocity_scale=20.0,
            normalize_loss=False,
            support="intrinsic",
            integrator="euler",
        )

    torch.manual_seed(seed)
    parameter = next(model.parameters())
    x_0 = flow.sample_prior(
        (n_samples, 2),
        device=device,
        dtype=parameter.dtype,
    )
    with torch.inference_mode():
        generated = flow.sample(model, x_0, n_steps=n_steps).cpu()
    metrics = checkerboard_distribution_metrics(generated)

    return {
        "method": method,
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "n_samples": n_samples,
        "n_steps": n_steps,
        "valid_tile_rate": float(metrics["valid_tile_rate"]),
        "histogram_tv": float(metrics["histogram_tv"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("rfm", "rgvfm"), default="rfm")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--n-samples", type=int, default=4_096)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint or find_checkpoint(project_root, args.method)
    result = evaluate_checkpoint(
        checkpoint,
        method=args.method,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
