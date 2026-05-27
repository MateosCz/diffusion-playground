import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from torch_geometric.data import Data, Batch
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
    def __init__(self, directory, size: int | None = None, seed: int| None = None):
        self.directory = directory
        self.data = torch.tensor(np.load(directory)) # (num_points, 2)
        self.data_scale = self._get_data_scale()
        self.data = self._normalize_data(self.data)
        self.seed = seed
        self.dataset_size = size
    
    def __len__(self):
        if self.dataset_size is not None:
            return self.dataset_size
        else:
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


class Shapes_Dataset(Dataset):
    """
    2D geometric shapes dataset. Each sample is a set of points uniformly
    sampled on the boundary of a random shape (triangle, rectangle, circle, or star).
    Each shape is randomly scaled, rotated, and translated, then normalized to [0, 1)^2.
 
    One sample = one shape = one group of points that move together in diffusion.
 
    Parameters:
        - num_points: number of points sampled on each shape's boundary.
        - dataset_size: virtual dataset length (for DataLoader compatibility).
        - shape_types: list of shape names to sample from.
          Options: 'triangle', 'rectangle', 'circle', 'star'
        - scale_range: (min_scale, max_scale) for random scaling.
        - seed: optional RNG seed for reproducibility.
    
    Returns:
        Tensor of shape (num_points, 2) with coordinates in [0, 1).
    """
 
    SHAPE_TYPES = ['triangle', 'rectangle', 'circle', 'star']
 
    def __init__(
        self,
        num_points: int = 64,
        dataset_size: int = 10000,
        shape_types: list[str] | None = None,
        scale_range: tuple[float, float] = (0.2, 0.8),
        seed: int | None = None,
    ):
        self.num_points = num_points
        self.dataset_size = dataset_size
        self.shape_types = shape_types or self.SHAPE_TYPES
        self.scale_range = scale_range
        self.seed = seed
 
    def __len__(self):
        return self.dataset_size
 
    def __getitem__(self, idx):
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + idx)
        else:
            generator = None
        return self._generate_shape_sample(generator)  # (points, corner_mask)
 
    # ------------------------------------------------------------------ #
    #  shape primitives — unit-scale, centered at origin                  #
    # ------------------------------------------------------------------ #
 
    @staticmethod
    def _unit_triangle(num_points: int, generator: torch.Generator = None):
        """Equilateral triangle with circumradius 1, centered at origin."""
        vertices = torch.stack([
            torch.tensor([math.cos(math.pi / 2 + 2 * math.pi * k / 3),
                          math.sin(math.pi / 2 + 2 * math.pi * k / 3)])
            for k in range(3)
        ])  # (3, 2)
        return _sample_polygon_edges(vertices, num_points, generator)
 
    @staticmethod
    def _unit_rectangle(num_points: int, generator: torch.Generator = None):
        """Unit-area rectangle (aspect ratio randomly between 0.5 and 2.0)."""
        # fixed aspect for the primitive; randomness comes from scale/rotate
        vertices = torch.tensor([
            [-1.0, -0.6],
            [ 1.0, -0.6],
            [ 1.0,  0.6],
            [-1.0,  0.6],
        ])
        return _sample_polygon_edges(vertices, num_points, generator)
 
    @staticmethod
    def _unit_circle(num_points: int, generator: torch.Generator = None):
        """Circle of radius 1 centered at origin."""
        if generator is None:
            jitter = torch.rand(num_points)
        else:
            jitter = torch.rand(num_points, generator=generator)
        # stratified samples on [0, 1) so small n still covers the full circle
        t = (torch.arange(num_points, dtype=torch.float32) + jitter) / num_points
        angles = t * 2 * math.pi
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        corner_mask = torch.zeros(num_points, dtype=torch.bool)
        return points, corner_mask  # (num_points, 2), (num_points,)
 
    @staticmethod
    def _unit_star(num_points: int, generator: torch.Generator = None, n_tips: int = 5):
        """
        5-pointed star with outer radius 1 and inner radius 0.4.
        Constructed as a 10-gon with alternating radii.
        """
        r_outer, r_inner = 1.0, 0.4
        angles = [math.pi / 2 + math.pi * k / n_tips for k in range(2 * n_tips)]
        radii = [r_outer if k % 2 == 0 else r_inner for k in range(2 * n_tips)]
        vertices = torch.tensor([
            [r * math.cos(a), r * math.sin(a)] for r, a in zip(radii, angles)
        ])
        return _sample_polygon_edges(vertices, num_points, generator)
 
    # ------------------------------------------------------------------ #
    #  sample generation                                                   #
    # ------------------------------------------------------------------ #
 
    def _generate_shape_sample(self, generator: torch.Generator = None):
        """
        Generate one sample: pick a random shape, apply random scale + rotation
        + translation, then normalize all points into [0, 1)^2.

        Returns:
            points:      (num_points, 2) with coordinates in [0, 1).
            corner_mask: (num_points,) bool — True for polygon corner vertices,
                         all-False for circles.
        """
        # --- pick shape type ---
        if generator is None:
            shape_idx = torch.randint(0, len(self.shape_types), (1,)).item()
        else:
            shape_idx = torch.randint(0, len(self.shape_types), (1,),
                                      generator=generator).item()
        shape_name = self.shape_types[shape_idx]

        # --- generate unit shape ---
        shape_fn = {
            'triangle':  self._unit_triangle,
            'rectangle': self._unit_rectangle,
            'circle':    self._unit_circle,
            'star':      self._unit_star,
        }[shape_name]
        points, corner_mask = shape_fn(self.num_points, generator)  # (N,2), (N,)

        # --- random rotation (apply before scaling so shape proportions are preserved) ---
        if generator is None:
            theta = torch.rand(1) * 2 * math.pi
        else:
            theta = torch.rand(1, generator=generator) * 2 * math.pi
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        R = torch.tensor([[cos_t, -sin_t],
                          [sin_t,  cos_t]]).squeeze()  # (2, 2)
        points = points @ R.T

        # --- normalize to [0,1)^2 via rescaling (not clamping) ---
        # 1) shift so that min corner is at origin
        p_min = points.min(dim=0).values  # (2,)
        p_max = points.max(dim=0).values  # (2,)
        points = points - p_min  # now in [0, bbox_w] x [0, bbox_h]

        # 2) scale longest side to target_size ∈ [scale_lo, scale_hi]
        bbox_span = (p_max - p_min).max()  # longest side of bounding box
        s_lo, s_hi = self.scale_range
        if generator is None:
            target_size = torch.rand(1) * (s_hi - s_lo) + s_lo
        else:
            target_size = torch.rand(1, generator=generator) * (s_hi - s_lo) + s_lo
        points = points / bbox_span * target_size  # longest side = target_size

        # 3) random translation within the remaining room
        actual_span = points.max(dim=0).values  # (2,) each ≤ target_size
        room_x = 1.0 - actual_span[0]  # available room for shifting in x
        room_y = 1.0 - actual_span[1]  # available room for shifting in y
        if generator is None:
            tx = torch.rand(1) * room_x
            ty = torch.rand(1) * room_y
        else:
            tx = torch.rand(1, generator=generator) * room_x
            ty = torch.rand(1, generator=generator) * room_y
        points[:, 0] += tx
        points[:, 1] += ty

        return points, corner_mask  # (num_points, 2), (num_points,)
 
 
# ---------------------------------------------------------------------- #
#  helper: uniform sampling on polygon edges                              #
# ---------------------------------------------------------------------- #
 
def _sample_polygon_edges(
    vertices: torch.Tensor,
    num_points: int,
    generator: torch.Generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `num_points` on the edges of a closed polygon, always including
    every corner vertex.  The remaining slots are filled with stratified
    uniform samples along the boundary.  All points are returned sorted by
    arc-length so they remain in boundary order.

    Args:
        vertices: (V, 2) ordered polygon vertices (closed automatically).
        num_points: number of points to sample (must be >= V).
        generator: optional torch RNG.

    Returns:
        points:      (num_points, 2) in boundary order.
        corner_mask: (num_points,) bool — True for the V corner vertices.
    """
    V = vertices.shape[0]
    next_v = torch.roll(vertices, -1, dims=0)
    edge_lengths = torch.norm(next_v - vertices, dim=-1)  # (V,)
    total_length = edge_lengths.sum()

    cum_lengths = torch.cumsum(edge_lengths, dim=0)  # (V,)
    cum_probs = cum_lengths / total_length            # (V,)

    # arc-length parameters of the corner vertices (vertex k sits at the
    # start of edge k, i.e. just after edge k-1 has ended)
    vertex_u = torch.cat([torch.zeros(1), cum_probs[:-1]])  # (V,)

    n_extra = max(0, num_points - V)

    if n_extra > 0:
        # stratified samples on [0, 1) for the non-corner slots
        if generator is None:
            jitter = torch.rand(n_extra)
        else:
            jitter = torch.rand(n_extra, generator=generator)
        u_extra = (torch.arange(n_extra, dtype=torch.float32) + jitter) / n_extra

        edge_idx = torch.searchsorted(cum_probs, u_extra).clamp(0, V - 1)

        lower = torch.zeros_like(cum_probs)
        lower[1:] = cum_probs[:-1]
        t_local = (
            (u_extra - lower[edge_idx])
            / (cum_probs[edge_idx] - lower[edge_idx] + 1e-12)
        ).unsqueeze(-1)

        p0 = vertices[edge_idx]
        p1 = next_v[edge_idx]
        extra_points = p0 + t_local * (p1 - p0)  # (n_extra, 2)

        all_u = torch.cat([vertex_u, u_extra])           # (num_points,)
        all_points = torch.cat([vertices, extra_points])  # (num_points, 2)
        all_mask = torch.cat([
            torch.ones(V, dtype=torch.bool),
            torch.zeros(n_extra, dtype=torch.bool),
        ])
    else:
        # fewer requested points than corners — return the first num_points vertices
        all_u = vertex_u[:num_points]
        all_points = vertices[:num_points]
        all_mask = torch.ones(num_points, dtype=torch.bool)

    # sort by arc-length to keep boundary order
    sort_idx = torch.argsort(all_u)
    return all_points[sort_idx], all_mask[sort_idx]

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
        result = self.base[idx]
        points = result[0] if isinstance(result, tuple) else result  # (num_points, 2) in [0, 1)
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
        result = self.base[idx]
        matrices = result[0] if isinstance(result, tuple) else result  # (dim, 2, 2)
        angles = torch.atan2(matrices[...,0,1], matrices[...,0,0])
        return angles # (dim,)

class PyGGraphWrapper(Dataset):
    """
    Wraps a point-cloud dataset into a PyG graph dataset.
 
    Each sample from the base dataset is (num_points, 2) in [0, 1).
    This wrapper converts it to a torch_geometric.data.Data with:
      - data.x:          (N, node_feat_dim) node features (angles + optional Fourier)
      - data.pos:        (N, 2) raw fractional coordinates
      - data.edge_index: (2, N*(N-1)) fully-connected edges (no self-loops)
      - data.edge_attr:  (N*(N-1), edge_feat_dim) pairwise angular differences
 
    Parameters:
        - base_dataset: any dataset returning (num_points, 2) tensors in [0, 1).
        - num_points_range: if given as (lo, hi), randomly vary num_points per sample
          by subsampling. If None, use all points from the base dataset.
        - use_fourier: if True, node features include [sin(θ1), cos(θ1), sin(θ2), cos(θ2)]
          in addition to raw angles. Recommended for periodic data.
        - seed: optional RNG seed for subsampling reproducibility.
    """
 
    def __init__(
        self,
        base_dataset: Dataset,
        num_points_range: tuple[int, int] | None = None,
        # use_fourier: bool = False,
        seed: int | None = None,
    ):
        self.base = base_dataset
        self.num_points_range = num_points_range
        # self.use_fourier = use_fourier
        self.seed = seed
 
    def __len__(self):
        return len(self.base)
 
    def __getitem__(self, idx):
        result = self.base[idx]
        if isinstance(result, tuple):
            points, corner_mask = result   # (N, 2), (N,) bool
        else:
            points = result
            corner_mask = torch.zeros(points.shape[0], dtype=torch.bool)

        # --- optional: subsample to variable point count ---
        if self.num_points_range is not None:
            lo, hi = self.num_points_range
            if self.seed is not None:
                gen = torch.Generator().manual_seed(self.seed + idx)
            else:
                gen = None

            n_corners = int(corner_mask.sum().item())
            n = torch.randint(lo, hi + 1, (1,), generator=gen).item()
            n = min(n, points.shape[0])   # can't exceed available points
            n = max(n, n_corners)         # must keep all corners

            corner_idx = corner_mask.nonzero(as_tuple=False).squeeze(-1)   # (n_corners,)
            non_corner_idx = (~corner_mask).nonzero(as_tuple=False).squeeze(-1)  # (N-n_corners,)

            n_extra = min(n - n_corners, non_corner_idx.shape[0])

            if n_extra > 0:
                # stratified subsample from non-corner boundary points
                if gen is not None:
                    jitter = torch.rand(n_extra, generator=gen)
                else:
                    jitter = torch.rand(n_extra)
                sample_pos = (
                    (torch.arange(n_extra, dtype=torch.float32) + jitter)
                    / n_extra
                    * non_corner_idx.shape[0]
                ).long().clamp(0, non_corner_idx.shape[0] - 1)
                extra_idx = non_corner_idx[sample_pos]
                selected = torch.cat([corner_idx, extra_idx])
            else:
                selected = corner_idx

            # restore boundary order
            selected = selected.sort().values
            points = points[selected]
            corner_mask = corner_mask[selected]
 
        N = points.shape[0]
 
        # --- node features: convert to periodic angles ---
        pos_in_theta = pos_to_angle(points)  # (N, 2) in [-pi, pi)
 
        # if self.use_fourier:
        #     # [θ1, θ2, sin(θ1), cos(θ1), sin(θ2), cos(θ2)]
        #     node_feat = torch.cat([
        #         pos_in_theta,
        #         torch.sin(pos_in_theta),
        #         torch.cos(pos_in_theta),
        #     ], dim=-1)  # (N, 6)
        # else:
        node_feat = pos_in_theta  # (N, 2)
 
        # --- fully-connected edge_index (no self-loops) ---
        row = torch.arange(N).repeat_interleave(N - 1)
        # for each node i, connect to all j != i
        col_parts = []
        for i in range(N):
            col_parts.append(torch.cat([torch.arange(0, i), torch.arange(i + 1, N)]))
        col = torch.cat(col_parts)
        edge_index = torch.stack([row, col], dim=0)  # (2, N*(N-1))
 
        # --- edge attributes: pairwise angular differences (wrapped) ---
        diff = pos_in_theta[col] - pos_in_theta[row]  # (E, 2) raw diff
        # wrap to [-pi, pi)
        # diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        # edge_attr = torch.cat([
        #     diff,
        #     torch.sin(diff),
        #     torch.cos(diff),
        # ], dim=-1)  # (E, 6)
        edge_attr = diff
 
        data = Data(
            x=node_feat,
            pos=pos_in_theta,
            edge_index=edge_index,
            edge_attr=edge_attr,
            corner_mask=corner_mask,
        )
        return data