import torch
import trimesh
import skimage
import numpy as np

from typing import Tuple
from deepSDF.config import Configuration
from deepSDF.data.data_creator import DataCreator


class ReconstructorHelper:
    @staticmethod
    def get_volume_coords(resolution: int = 256, device: str = Configuration.DEVICE) -> Tuple[torch.Tensor, int]:
        # https://github.com/maurock/DeepSDF/blob/main/utils/utils_deepsdf.py#L51-L62

        grid_values = torch.arange(-1, 1, float(1 / resolution)).to(device)
        grid = torch.meshgrid(grid_values, grid_values, grid_values)

        grid_size_axis = grid_values.shape[0]

        coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

        return coords, grid_size_axis

    @staticmethod
    def extract_mesh(grid_size_axis: int, sdf: torch.Tensor, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # https://github.com/maurock/DeepSDF/blob/main/utils/utils_deepsdf.py#L84-L94

        grid_sdf = sdf.reshape(grid_size_axis, grid_size_axis, grid_size_axis).detach().cpu().numpy()

        if not (grid_sdf.min() <= 0.00 <= grid_sdf.max()):
            return None, None

        vertices, faces, _, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

        if normalize:
            x_max = np.array([1, 1, 1])
            x_min = np.array([-1, -1, -1])
            vertices = vertices * ((x_max - x_min) / grid_size_axis) + x_min

        return vertices, faces


class Reconstructor(ReconstructorHelper):
    def __init__(self, model, resolution):
        self.model = model
        self.resolution = resolution


if __name__ == "__main__":
    from debugvisualizer.debugvisualizer import Plotter

    mesh = DataCreator.load_mesh(r"deepSDF\data\raw\0.obj", normalize=True, map_z_to_y=True, check_watertight=True)

    coords, grid_size_axis = ReconstructorHelper.get_volume_coords(resolution=130)

    sdf = DataCreator.compute_sdf(mesh, coords.cpu().numpy(), sigma=0)
    sdf = torch.tensor(sdf).unsqueeze(dim=1).to(Configuration.DEVICE)

    vertices, faces = ReconstructorHelper.extract_mesh(grid_size_axis, sdf)

    mesh_by_sdf = trimesh.Trimesh(vertices, faces)

    Plotter(mesh_by_sdf, mesh, map_z_to_y=False).save()
