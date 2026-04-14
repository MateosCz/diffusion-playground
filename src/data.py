from ast import Tuple
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
checkerboard groundtruth plot
"""

        


"""
Pac-man data generation
"""