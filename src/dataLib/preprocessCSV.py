"""
preprocess the material csv file
not all the columns are useful, so we need to preprocess the csv file
and extract the useful information
input: csv file
output: pytorch geometric data object
"""

import os
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from tqdm import tqdm

from src.dataLib.data_util import save_json


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """

    def abs_cap(val: float, max_abs_val: float = 1.0):
        return max(min(val, max_abs_val), -max_abs_val)

    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def process_cif(crystal_str: str) -> dict[str, np.ndarray]:
    crystal = Structure.from_str(crystal_str, fmt="cif")
    crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )

    frac_coords = canonical_crystal.frac_coords
    atom_types = canonical_crystal.atomic_numbers
    lattice_parameters = canonical_crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(
        canonical_crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
    ) # check if the lengths and angles are correct

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)

    return {"pos": frac_coords, "h": atom_types, "lengths": lengths, "angles": angles}

def preprocess_matcsv(csv_folder: str,
    output_folder: str = None,
    splits: Literal["train", "val", "test"] = "train",
    max_atoms: int = -1
):
    """
    preprocess the csv file
    """
    for split in tqdm(splits):
        csv_file = os.path.join(csv_folder, f"{split}.csv")
        df = pd.read_csv(csv_file, index_col=0)
        n_structures = len(df) # number of structures(materials or samples) in the csv file
        data_list = []
        mean_std_dict = dict()
        for i in tqdm(range(n_structures), desc=f"Processing {split} split"):
            cif = df.iloc[i]["cif"]
            cif_info = process_cif(cif)
            num_atoms = len(cif_info["pos"])
            if max_atoms > 0 and num_atoms > max_atoms:
                continue # skip if the number of atoms is greater than max_atoms

            # store the mean and std of the lengths
            if num_atoms not in mean_std_dict:
                mean_std_dict[num_atoms] = [] # if the number of atoms is not in the dictionary, initialize it as an empty list
            mean_std_dict[num_atoms].append(cif_info["lengths"]) # store the lengths of the structure

            data = Data(
                pos = torch.tensor(cif_info["pos"]),
                h = torch.tensor(cif_info["h"]),
                lengths = torch.tensor(cif_info["lengths"]).view(1, -1), #  Å
                angles = torch.tensor(cif_info["angles"]).view(1, -1) # degrees
            )
            data_list.append(data)
        # save path is either the output folder or the csv folder
        save_path = os.path.join(output_folder, f"{split}.pt") if output_folder is not None else os.path.join(csv_folder, f"{split}.pt")
        torch.save(data_list, save_path)
        print(f"Saved {len(data_list)} structures to {save_path}, preprocessed {split} split")

        # compute stats, using log scaling
    
        log_mean_std_dict = { 
            n: np.log(np.array(mean_std_dict[n])) for n in mean_std_dict
        }

        mean_std_stats = {}
        for n in log_mean_std_dict:
            log_abc_ind_sorted = np.sort(log_mean_std_dict[n], axis=0) # sort the log abc lengths independently
            idx = int(len(log_abc_ind_sorted) * 0.025) # compute mean and std between the 2.5% and 97.5% quantiles
            log_abc_quantiles = log_abc_ind_sorted[idx: -idx, :]
            log_abc_mean = np.mean(log_abc_quantiles, axis=0)
            log_abc_std = np.std(log_abc_quantiles, axis=0)
            mean_std_stats[n] = (log_abc_mean.tolist(), log_abc_std.tolist())
        save_path = os.path.join(output_folder, f"{split}_mean_std_stats.json") if output_folder is not None else os.path.join(csv_folder, f"{split}_mean_std_stats.json")
        save_json(json_dict=mean_std_stats, json_path=save_path, sort_keys=True)
        print(f"Saved {split} mean and std stats to {save_path}")
    