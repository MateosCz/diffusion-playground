from typing import Callable, Optional, Sequence

import chemparse as chemparse
import numpy as np
import torch
import torch.utils.data as torchdata
from ase.data import atomic_numbers
from torch_geometric.data import Data
from torch_geometric.io import fs



class CrystalDataset(torchdata.Dataset):
    def __init__(
            self,
            path: str,
            transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.data = self.load(path)

    @staticmethod
    def load(path):
        return fs.torch_load(path)

    def __getitem__(self, idx):
        data = self.data[idx]

        if not isinstance(data, Data):  # the data was saved as a dict of numpy arrays
            data = Data(
                pos=torch.Tensor(data["pos"]), # pos is the fractional positions
                h=torch.LongTensor(data["h"]), # h is the atom types
                lengths=torch.Tensor(data["lengths"]).view(1, -1), # lengths is the lattice lengths
                angles=torch.Tensor(data["angles"]).view(1, -1), # angles is the lattice angles
            )

        data = data if self.transform is None else self.transform(data)
        data.x = data.pos # to fit the TDM model input format

        return data

    def __len__(self):
        return len(self.data)


"""
dataset for CSP evaluation and generation
input: a list of formulas
output: a list of pytorch geometric data objects
"""

class SampleDatasetCSP(torchdata.Dataset):
    
    def __init__(
            self,
            formulas: Sequence[str],
            n_samples_per_formula: int = 5,
            transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.parsed_formulas = [
            chemparse.parse_formula(f)
            for f in formulas
            for _ in range(n_samples_per_formula)
        ]

    def __getitem__(self, idx):
        formula: dict[str, float] = self.parsed_formulas[idx]
        h = torch.LongTensor(
            [atomic_numbers[elem] for elem in formula for _ in range(int(formula[elem]))]
        )

        data = Data(
            pos=torch.randn(len(h), 3),
            h=h,
        )
        data = data if self.transform is None else self.transform(data)
        data.x = data.pos # to fit the TDM model input format

        return data

    def __len__(self):
        return len(self.parsed_formulas)
