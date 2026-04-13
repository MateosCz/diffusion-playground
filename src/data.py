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
    get the num of tiles, num of sampled points, and the optional Lie data
    return the sampled points fractional coordinates.

    parameters:
        - num_rows, the counts of rows of tiles.
        - num_points, the counts of points of each sample
        - isLie, Bool type, whether sampled as Lie data

    """
    def __init__(self, num_rows, num_points, isLie):
        self.num_rows = num_rows
        self.num_points = num_points
        self.isLie = isLie
        


"""
Pac-man data generation
"""