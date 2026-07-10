"""
Preprocess a crystal material dataset from CSV splits into PyG ``.pt`` files.

Each dataset lives under ``data/<dataset>/`` and contains ``train.csv``,
``val.csv`` and ``test.csv`` (each with a ``cif`` column). For every split this
script produces ``<split>.pt`` (a list of PyG ``Data`` objects) and
``<split>_mean_std_stats.json`` (per-atom-count lattice-length statistics).

Example:
    python -m src.scripts.preprocess --dataset perov-5
    python -m src.scripts.preprocess --dataset mp_20 --splits train val --max-atoms 20
"""

import argparse
import os

from src.dataLib.preprocessCSV import preprocess_matcsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a crystal dataset (CSV -> PyG .pt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset name, i.e. the sub-folder under --data-root (e.g. perov-5, mp_20, carbon-24).",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory that contains the dataset folder.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Where to write the .pt / stats files. Defaults to the dataset folder.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to preprocess.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=-1,
        help="Skip structures with more than this many atoms (-1 keeps all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_folder = os.path.join(args.data_root, args.dataset)
    if not os.path.isdir(csv_folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {csv_folder!r}. "
            f"Expected a directory under {args.data_root!r} named after the dataset."
        )

    if args.output_folder is not None:
        os.makedirs(args.output_folder, exist_ok=True)

    print(f"Preprocessing dataset '{args.dataset}' from {csv_folder}")
    print(f"Splits: {args.splits} | max_atoms: {args.max_atoms}")

    preprocess_matcsv(
        csv_folder=csv_folder,
        output_folder=args.output_folder,
        splits=args.splits,
        max_atoms=args.max_atoms,
    )

    print("Done.")


if __name__ == "__main__":
    main()
