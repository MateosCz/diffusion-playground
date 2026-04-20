import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
    """Map theta to x \in [a,b)"""
    x = (theta/ (2*torch.pi)+0.5) * (b-a)
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

def wrap_fractional(x, x_range= (2.0*torch.pi)):
    wrapped_x = torch.arctan2(torch.sin(x_range * x), torch.cos(x_range * x)) / x_range
    return wrapped_x

def wrap_angle(x):
    wrapped_x = torch.arctan2(torch.sin(x), torch.cos(x))
    return wrapped_x






"""
data generation
"""


"""
checkerboard data generation
"""

class Checkerboard_Dataset(Dataset):
    """
    Square checkerboard dataset, num of tiles should be (num_row * num_row)
    get the num of tiles, num of sampled points, num of total samples
    return the sampled points fractional coordinates [0,1).

    parameters:
        - num_rows: the counts of rows of tiles.
        - num_points: the counts of points of each sample
        - dataset_size: Virtual length of the dataset (for DataLoader compatibility). 
        Since we generate on the fly, the value is somewhat arbitrary. 
        E.g. DataLoader iterates dataset_size / batch_size steps per epoch. So 10_000 with batch_size=32 gives ~312 steps per epoch.

    """
    def __init__(self, num_rows, num_points, dataset_size = 10000, oversample_factor = 2.5, seed: int| None = None):
        self.num_rows = num_rows
        self.num_points = num_points
        self.dataset_size = dataset_size
        self.oversample_factor = oversample_factor
        self.seed = seed

    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self,idx):
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + idx)
            return self._generate_checkerboard_sample(
                self.num_rows, 
                self.num_points, 
                self.oversample_factor,
                generator)
        else:
            return self._generate_checkerboard_sample(
                self.num_rows,
                self.num_points,
                self.oversample_factor
            )


    def _generate_checkerboard_sample(
        self,
        num_rows: int,
        num_points: int,
        oversample_factor: float = 2.5,
        generator: torch.Generator = None
    ):
        """
        Generate one sample of `num_points` on the black tiles of an num_rows x num_rows checkerboard.

        Args:
            - num_rows: Number of tile rows (and columns). Must be >= 1.
            - num_points: Number of accepted points per sample.
            - oversample_factor: How many candidates to propose per expected accept.
            Acceptance rate is ~0.5, so 2.5 gives a comfortable margin.
            - generator: RNG state used for torch.rand(). Passing a per-index generator
            (seeded with self.seed + idx) ensures each sample is reproducible
            while remaining independent across indices.
        
        Returns:
            Tensor of shape (n_points, 2) with coordinates in [0, 1).
        """
        collected = []
        remaining = num_points

        while remaining > 0:
            # propose candidates uniformly in [0,1)**2
            num_candidates = int(remaining * oversample_factor) + 1
            if generator is None:
                candidates = torch.rand(num_candidates, 2) # (N, 2)
            else:
                candidates = torch.rand(num_candidates, 2, generator=generator)

            # Tile indices for each candidate
            tile_x = (candidates[:,0] * num_rows).long()
            tile_y = (candidates[:,1] * num_rows).long()

            on_black = ((tile_x + tile_y) % 2 ) ==0
            accepted = candidates[on_black]

            collected.append(accepted[:remaining])
            remaining -= len(collected[-1])

        return torch.cat(collected, dim=0) # (n_points, 2)


"""
Lie torus dataset wrapper

"""

class TorusLieWrapper(Dataset):
    """
    Wraps any dataset that returns (num_points, 2) tensors in [0, 1)
    and converts them to SO(2) x SO(2) rotation matrices.
    
    Output shape: (num_points, 2, 2, 2) — num_points pairs of 2x2 rotation matrices.    
    """

    def __init__(self, base_dataset):
        self.base = base_dataset
    
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        points = self.base[idx]                          # (num_points, 2) in [0, 1)
        angles = (points - 0.5) * 2 * torch.pi          # (num_points, 2) in [-pi, pi)
        c, s = torch.cos(angles), torch.sin(angles)
        row0 = torch.stack([c, s], dim=-1)               # (num_points, 2, 2 (row1))
        row1 = torch.stack([-s, c], dim=-1)
        matrices = torch.stack([row0, row1], dim=-2)     # (num_points, 2(theta1,theta2, corresponding to x,y), 2(row1), 2(row2))
        return matrices




"""
Pac-man data generation
"""