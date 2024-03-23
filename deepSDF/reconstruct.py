import torch
import trimesh
import skimage
import numpy as np

from tqdm import tqdm
from typing import Tuple
from deepSDF.config import Configuration


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


class Reconstructor(ReconstructorHelper, Configuration):
    def __init__(self):
        pass

    def reconstruct(self, sdf_decoder, sdf_dataset, obj_path, epoch) -> None:
        """_summary_

        Args:
            sdf_decoder (_type_): _description_
            sdf_dataset (_type_): _description_
            obj_path (_type_): _description_
            epoch (_type_): _description_
        """

        sdf_decoder.eval()
        with torch.no_grad():
            coords, grid_size_axis = self.get_volume_coords(resolution=int(self.RECONSTRUCT_RESOLUTION))
            coords.to(self.DEVICE)
            coords_batches = torch.split(coords, coords.shape[0] // 1000)

            sdf = torch.tensor([]).to(self.DEVICE)

            cls_rand = int(torch.randint(low=0, high=sdf_dataset.cls_nums, size=(1,)))

            for coords_batch in tqdm(coords_batches, desc=f"Reconstructing in `{epoch}th` epoch", leave=False):
                cls = torch.tensor([cls_rand] * coords_batch.shape[0], dtype=torch.long).to(self.DEVICE)
                pred = sdf_decoder(cls, coords_batch)
                if sum(sdf.shape) == 0:
                    sdf = pred
                else:
                    sdf = torch.vstack([sdf, pred])

            vertices, faces = self.extract_mesh(grid_size_axis=grid_size_axis, sdf=sdf)

            if vertices is not None and faces is not None:
                cls_name = sdf_dataset.cls_dict[cls_rand]

                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
                mesh.export(obj_path.replace(".obj", f"_{epoch}_{cls_name}.obj"))

        sdf_decoder.train()
