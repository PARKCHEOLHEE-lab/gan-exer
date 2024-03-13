import torch
import skimage
import numpy as np

from deepSDF.config import Configuration
from typing import Tuple


class ReconstructorHelper(Configuration):
    def __init__(self):
        pass

    def get_volume_coords(self, resolution: int = 256) -> Tuple[torch.Tensor, int]:
        grid_values = torch.arange(-1, 1, float(1 / resolution)).to(self.DEVICE)
        grid = torch.meshgrid(grid_values, grid_values, grid_values)

        grid_size_axis = grid_values.shape[0]

        coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(self.DEVICE)

        return coords, grid_size_axis

    def extract_mesh(self, grid_size_axis, sdf):
        # Extract zero-level set with marching cubes
        grid_sdf = sdf.reshape(grid_size_axis, grid_size_axis, grid_size_axis)
        vertices, faces, _, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        vertices = vertices * ((x_max - x_min) / grid_size_axis) + x_min

        return vertices, faces


class Reconstructor(Configuration):
    def __init__(self, model):
        self.model = model
