import pytest
import torch

from src.lit.checkerboard_generation_metrics import (
    checkerboard_distribution_metrics,
)


def test_checkerboard_metrics_are_ideal_for_uniform_valid_bins():
    bins = 16
    centers = (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
    grid_x, grid_y = torch.meshgrid(centers, centers, indexing="ij")
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    tile = torch.floor(points * 4).long()
    points = points[(tile[:, 0] + tile[:, 1]).remainder(2) == 0]

    metrics = checkerboard_distribution_metrics(points, bins=bins)

    torch.testing.assert_close(metrics["valid_tile_rate"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["histogram_tv"], torch.tensor(0.0))


def test_checkerboard_metrics_detect_uniform_torus_baseline():
    bins = 16
    centers = (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
    grid_x, grid_y = torch.meshgrid(centers, centers, indexing="ij")
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    metrics = checkerboard_distribution_metrics(points, bins=bins)

    torch.testing.assert_close(metrics["valid_tile_rate"], torch.tensor(0.5))
    torch.testing.assert_close(metrics["histogram_tv"], torch.tensor(0.5))


def test_checkerboard_metrics_validate_inputs():
    with pytest.raises(ValueError):
        checkerboard_distribution_metrics(torch.rand(4, 3))
    with pytest.raises(ValueError):
        checkerboard_distribution_metrics(torch.rand(4, 2), bins=15)
