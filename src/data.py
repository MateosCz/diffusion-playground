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
        - dataset_size: Virtual length of the dataset (for DataLoader compatibility). 
        Since we generate on the fly, the value is somewhat arbitrary. 
        E.g. DataLoader iterates dataset_size / batch_size steps per epoch. So 10_000 with batch_size=32 gives ~312 steps per epoch.

    """
    def __init__(self, num_rows, dataset_size = 10000, seed: int| None = None):
        self.num_rows = num_rows
        self.dataset_size = dataset_size
        self.seed = seed

    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self,idx):
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + idx)
            return self._generate_checkerboard_sample(
                self.num_rows, 
                generator)
        else:
            return self._generate_checkerboard_sample(
                self.num_rows,
            )


    def _generate_checkerboard_sample(
        self,
        num_rows: int,
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

        while True:
            # propose candidates uniformly in [0,1)**2
            if generator is None:
                point = torch.rand(2) # (2,)
            else:
                point = torch.rand(2, generator=generator)

            # Tile indices for each candidate
            tile_x = (point[0] * num_rows).long()
            tile_y = (point[1] * num_rows).long()
            if ((tile_x + tile_y) % 2) == 0:
                return point  # (2,)


class Pacman_Dataset(Dataset):
    """
    Pacman maze dataset, uniformly sample from the maze .npy document.
    directory: data/pacman.npy
    """
    def __init__(self, directory, seed: int| None = None):
        self.directory = directory
        self.data = torch.tensor(np.load(directory)) # (num_points, 2)
        self.data_scale = self._get_data_scale()
        self.data = self._normalize_data(self.data)
        self.seed = seed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + idx)
            return self._generate_pacman_sample(
                generator)
        else:
            return self._generate_pacman_sample()


    def _generate_pacman_sample(self, generator: torch.Generator = None):
        """
        generate a single pacman sample from
        """
        if generator is None:
            rand_index = torch.randint(0, len(self.data), size=(1,))
        else:
            rand_index = torch.randint(0, len(self.data), size=(1,), generator=generator)
        return self.data[rand_index].squeeze(0)
    def _get_data_scale(self):
        """
        get the scale of the data
        """
        return torch.max(self.data) - torch.min(self.data)
    def _normalize_data(self, data):
        """
        normalize the data to [0, 1]
        """
        return (data - torch.min(self.data)) / self._get_data_scale()

"""
Lie torus dataset wrapper

"""

class TorusLieWrapper(Dataset):
    """
    Wraps any dataset that returns (num_points, 2) tensors in [0, 1)
    and converts them to SO(2) x SO(2) rotation matrices.
    
    Output shape: (2, 2, 2) — 2x2 rotation matrices.    
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


class AngleTorusWrapper(Dataset):
    """
    Wraps any dataset that returns rotation matrices in SO(2)
    and converts them to angle torus data in [-pi, pi)
    
    Output shape: (dim) — dim angles in [-pi, pi).    
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        matrices = self.base[idx]                          # (dim, 2, 2)
        angles = torch.atan2(matrices[...,0,1], matrices[...,0,0])
        return angles # (dim,)

"""
Pac-man data generation
"""