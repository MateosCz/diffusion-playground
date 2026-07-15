import torch
from torch_geometric.utils import scatter
from torch_geometric.data import Data


import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

import ase.io
import numpy as np
from ase.atoms import Atoms
from ase.data import chemical_symbols
from pymatgen.core import Lattice, Structure

if TYPE_CHECKING:
    from src.dataLib.transform import (
        ContinuousIntervalAngles,
        ContinuousIntervalLengths,
        Pos2Angle,
    )


"""
data processing helpers
"""

def so2mat_to_pos(so2matrix):
    """
    recover fractional coordinates from so2 matrix
    """
    theta = torch.sign(so2matrix[...,0,1]) * torch.arccos(so2matrix[...,0,0])
    x = (theta/(2 * torch.pi)+ 0.5) * (1-0)
    return x

def so2mat_to_angle(so2matrix):
    """
    recover theta(angle) from so2 matrix 
    """
    theta = torch.sign(so2matrix[...,0,1]) * torch.arccos(so2matrix[...,0,0])
    return theta

def angle_to_pos(theta, a=0, b=1):
    """Map theta from [-pi, pi] to x in [a, b]."""
    x = (theta / (2 * torch.pi) + 0.5) * (b - a) + a
    return x

def pos_to_angle(x, a=0,b=1):
    """
    Map x from [a,b) to theta [-pi,pi)
    """
    theta = 2 * torch.pi * (x - a) / (b - a) - torch.pi
    return theta

def theta_to_so2mat(theta):
    """Construct SO(2) representation g from θ."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    g = torch.stack([
        torch.stack([cos_t, sin_t], dim=-1),
        torch.stack([-sin_t, cos_t], dim=-1),
    ], dim=-2)
    return g

def torus_embedding(theta1: torch.Tensor, theta2: torch.Tensor, R:float = 3.0, r: float = 1.0):
    """construct the torus embedding for SO(2)xSO(2) data"""
    x = (R + r * torch.cos(theta2)) * torch.cos(theta1)
    y = (R + r * torch.cos(theta2)) * torch.sin(theta1)
    z = r * torch.sin(theta2)
    return torch.stack([x,y,z], dim=-1)

"""
Wrap periodic fractional-coordinate-like data from R to [-pi,pi]
"""
def wrap_pos(x, x_range:float = 1.0):
    wrapped_x = torch.arctan2(torch.sin(x_range * x), torch.cos(x_range * x)) / x_range
    return wrapped_x

"""wrap periodic angle-like data x from [theta,theta+2kpi](or R) to [-pi,pi]""" 
def wrap_angle(x):
    wrapped_x = torch.arctan2(torch.sin(x), torch.cos(x))
    return wrapped_x



def wrap_01(x):
    return torch.remainder(x, 1.0)   # equivalent to x % 1.0, maps to [0, 1)

def wrapped_diff(a, b):
    """Signed angular difference a - b, wrapped to [-pi, pi)."""
    return torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi

# scatter center the data by subgraph's mean
def scatter_center(x, idx):
    if idx.device != x.device:
        idx = idx.to(x.device)
    if idx.dtype != torch.long:
        idx = idx.long()
    centers = scatter(x, idx, dim=0, reduce="mean")
    centers = centers[idx]
    return x - centers


def read_json(json_path: str | Path):
    with open(json_path, encoding="utf-8", mode="r") as fp:
        return json.load(fp)


def save_json(json_dict: dict, json_path: str, sort_keys: bool = False):
    def _fix_dict():
        for key in json_dict:
            if isinstance(json_dict[key], np.ndarray):
                json_dict[key] = json_dict[key].tolist()

    _fix_dict()
    with open(json_path, encoding="utf-8", mode="w") as fp:
        json.dump(json_dict, fp, sort_keys=sort_keys)


def save_images(
        images: Sequence[ase.Atoms], filename: Union[str, Path], fmt: str = "extxyz"
):
    ase.io.write(filename=filename, images=images, format=fmt)

def PyGData_to_Structure(
    data: Data,
    transform_lengths=None,
    transform_angles=None,
    transform_positions=None,
):
    from src.dataLib.preprocessCSV import lattice_params_to_matrix
    from pymatgen.core import Lattice, Structure

    # The lattice is diffused as the concatenated vector `l` = [lengths, angles],
    # while `lengths`/`angles` keep their original (unnoised) values. Prefer `l`
    # so noised samples reflect the updated lattice; fall back to the split keys.
    if getattr(data, "l", None) is not None:
        l = torch.as_tensor(data.l).detach().cpu().view(-1)
        log_lengths, tan_angles = l[:3], l[3:]
    else:
        log_lengths = torch.as_tensor(data.lengths).detach().cpu().view(-1)
        tan_angles = torch.as_tensor(data.angles).detach().cpu().view(-1)

    if transform_lengths is not None:
        n_atoms = int(data.x.shape[0]) if getattr(data, "x", None) is not None else int(data.pos.shape[0])
        a, b, c = transform_lengths.invert_one(log_lengths.numpy(), n_atoms)
        lengths = np.array([a, b, c])
    else:
        lengths = torch.exp(log_lengths).numpy()

    if transform_angles is not None:
        alpha, beta, gamma = transform_angles.invert_one(tan_angles.numpy())
        angles = np.array([alpha, beta, gamma])
    else:
        angles = torch.rad2deg(torch.arctan(tan_angles) + torch.pi / 2.0).numpy()

    x = torch.as_tensor(data.x).detach().cpu()
    if transform_positions is not None:
        frac_coords = transform_positions.invert_one(x.numpy())
    else:
        frac_coords = angle_to_pos(wrap_angle(x)).numpy()

    numbers = torch.as_tensor(data.h).detach().cpu().numpy()
    # lattice = lattice_params_to_matrix(*lengths, *angles)

    return Structure(
        # lattice=Lattice(lattice),
        lattice=Lattice.from_parameters(a=a,b=b,c=c,alpha=alpha,beta=beta,gamma=gamma),
        species=numbers,
        coords=frac_coords,
        coords_are_cartesian=False,
    )

def kldm_output_to_structures_one(
    graph: Data,
    l0: torch.Tensor,
    f0: torch.Tensor,
    transform_lengths: "ContinuousIntervalLengths",
    transform_angles: "ContinuousIntervalAngles",
    transform_positions: "Pos2Angle",
):
    """Convert one reverse-diffusion output into a pymatgen Structure."""
    data = Data(
        x=torch.as_tensor(f0),
        h=graph.h,
        l=torch.as_tensor(l0).reshape(1, -1),
    )
    return PyGData_to_Structure(
        data,
        transform_lengths=transform_lengths,
        transform_angles=transform_angles,
        transform_positions=transform_positions,
    ).get_sorted_structure()


def kldm_output_to_structures_batch(
    graph: Data,
    l0: torch.Tensor,
    f0: torch.Tensor,
    transform_lengths: "ContinuousIntervalLengths",
    transform_angles: "ContinuousIntervalAngles",
    transform_positions: "Pos2Angle",
):
    """Split the batched reverse-diffusion output into pymatgen Structures."""
    batch_vec = graph.batch.detach().cpu()
    f0 = f0.detach().cpu()
    l0 = l0.detach().cpu()

    structures = []
    for g in range(int(batch_vec.max().item()) + 1):
        mask = batch_vec == g
        structure = kldm_output_to_structures_one(
            graph[g],
            l0[g],
            f0[mask],
            transform_lengths,
            transform_angles,
            transform_positions,
        )
        structures.append(structure)
    return structures


def PyGData_to_AseAtoms(data: Data, transform_lengths=None, transform_angles=None, transform_positions=None):
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = PyGData_to_Structure(
        data,
        transform_lengths=transform_lengths,
        transform_angles=transform_angles,
        transform_positions=transform_positions,
    )
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.pbc = True
    return atoms