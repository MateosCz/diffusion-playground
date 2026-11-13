"""Attach a training-time element encoding to a Lightning CSP checkpoint.

Use the original encoding artifact from the machine that trained the model.
This command never fits PCA or changes neural weights and writes a new file.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

import torch

from src.dataLib.type_encoding_state import export_type_encoding, restore_type_encoding


def migrate_checkpoint(checkpoint, encoding_state, *, expected_encoding_sha256=None):
    """Return a checkpoint with a validated original encoding and lookup buffers.

    The caller supplies the training artifact explicitly: matching dimensions
    alone cannot establish that an arbitrary PCA table belongs to these weights.
    Existing neural, EMA, optimizer and training-loop state is preserved.
    """
    encoding = restore_type_encoding(encoding_state)
    if not encoding.name.startswith("subatomic_tokenizer_"):
        raise ValueError("CSP encoding migration requires a subatomic tokenizer state.")
    state = export_type_encoding(encoding)
    if expected_encoding_sha256 is not None and state["state_sha256"] != expected_encoding_sha256:
        raise ValueError(
            "Encoding state hash does not match --expected-encoding-sha256: "
            f"got {state['state_sha256']}."
        )

    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("Expected a Lightning checkpoint containing state_dict.")
    weights = checkpoint["state_dict"]
    input_weight = weights.get("pi.atom_type_emb_mlp.0.weight")
    if not torch.is_tensor(input_weight) or input_weight.ndim != 2:
        raise ValueError("Checkpoint does not contain a token-based CSP model.")
    if input_weight.shape[1] != encoding.type_dim:
        raise ValueError(
            f"Encoding dimension {encoding.type_dim} does not match the model's "
            f"{input_weight.shape[1]} input channels."
        )

    embedded = checkpoint.get("type_encoding_state")
    if embedded is not None:
        existing = export_type_encoding(restore_type_encoding(embedded))
        if existing["state_sha256"] != state["state_sha256"]:
            raise ValueError("Supplied encoding conflicts with the checkpoint's embedded encoding.")

    new_weights = weights.copy()
    lookup = encoding.z_to_enc.detach().cpu()
    prefixes = ["pi."]
    if any(key.startswith("ema_pi.module.") for key in weights):
        prefixes.append("ema_pi.module.")
    for prefix in prefixes:
        key = f"{prefix}atom_type_table"
        if key in weights:
            current = weights[key]
            if (not torch.is_tensor(current) or current.dtype != lookup.dtype
                    or current.shape != lookup.shape
                    or not torch.equal(current.detach().cpu(), lookup)):
                raise ValueError(f"Checkpoint lookup buffer {key} conflicts with the encoding state.")
        else:
            new_weights[key] = lookup.clone()

    migrated = dict(checkpoint)
    migrated["state_dict"] = new_weights
    migrated["type_encoding_state"] = embedded if embedded is not None else state
    return migrated


def migrate_checkpoint_file(
    source, encoding_path, output, *, expected_encoding_sha256=None,
):
    """Write a new complete checkpoint atomically, without overwriting a file."""
    source, encoding_path, output = Path(source), Path(encoding_path), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Output must be a new file, separate from the source checkpoint.")
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    encoding_state = torch.load(encoding_path, map_location="cpu", weights_only=True)
    migrated = migrate_checkpoint(
        checkpoint, encoding_state, expected_encoding_sha256=expected_encoding_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp",
            dir=output.parent, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(migrated, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Linking publishes the completed file and fails if another writer has
        # created the destination since the initial existence check.
        os.link(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return migrated["type_encoding_state"]["state_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Original Lightning CSP checkpoint.")
    parser.add_argument(
        "--type-encoding-state", required=True,
        help="Original training-time subatomic tokenizer state (.pt).",
    )
    parser.add_argument("--output", required=True, help="New checkpoint path; must not exist.")
    parser.add_argument(
        "--expected-encoding-sha256", default=None,
        help="Require this canonical encoding state hash before writing.",
    )
    args = parser.parse_args()
    state_hash = migrate_checkpoint_file(
        args.ckpt, args.type_encoding_state, args.output,
        expected_encoding_sha256=args.expected_encoding_sha256,
    )
    print(f"Saved checkpoint: {args.output}")
    print(f"Encoding state SHA-256: {state_hash}")


if __name__ == "__main__":
    main()
