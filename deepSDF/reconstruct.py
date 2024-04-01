import time
import torch
import trimesh
import skimage
import numpy as np
import point_cloud_utils as pcu

from tqdm import tqdm
from typing import Tuple
from deepSDF.config import Configuration


class ReconstructorHelper:
    @staticmethod
    def get_volume_coords(resolution: int = 256, device: str = Configuration.DEVICE) -> Tuple[torch.Tensor, int]:
        # https://github.com/maurock/DeepSDF/blob/main/utils/utils_deepsdf.py#L51-L62

        grid_values = torch.arange(0, 1, float(1 / resolution)).to(device)
        grid = torch.meshgrid(grid_values, grid_values, grid_values)

        grid_size_axis = grid_values.shape[0]

        coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

        return coords, grid_size_axis

    @staticmethod
    def extract_mesh(
        grid_size_axis: int,
        sdf: torch.Tensor,
        normalize: bool = True,
        map_z_to_y: bool = False,
        check_watertight: bool = False,
    ) -> trimesh.Trimesh:
        # https://github.com/maurock/DeepSDF/blob/main/utils/utils_deepsdf.py#L84-L94

        grid_sdf = sdf.reshape(grid_size_axis, grid_size_axis, grid_size_axis).detach().cpu().numpy()

        if not (grid_sdf.min() <= 0.00 <= grid_sdf.max()):
            return None

        vertices, faces, _, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

        if normalize:
            x_max = np.array([1, 1, 1])
            x_min = np.array([0, 0, 0])
            vertices = vertices * ((x_max - x_min) / grid_size_axis) + x_min

        mesh = trimesh.Trimesh(vertices, faces)

        if check_watertight and not mesh.is_watertight:
            vertices, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=100000)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if map_z_to_y:
            mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

        return mesh


class Reconstructor(ReconstructorHelper, Configuration):
    def __init__(self):
        pass

    def reconstruct(
        self,
        sdf_decoder,
        cls_dict,
        obj_path,
        epoch,
        cls_num=None,
        resolution=Configuration.RECONSTRUCT_RESOLUTION,
        normalize=True,
        map_z_to_y=False,
    ) -> None:
        """Generate mesh from the predicted SDF values

        Args:
            sdf_decoder: model
            sdf_dataset: dataset
            obj_path: path to save the reconstructed mesh
            epoch: epoch
            cls_num: class number
            resolution: resolution for reconstructing
            normalize: normalize
            map_z_to_y: map z to y
        """

        sdf_decoder.eval()
        with torch.no_grad():
            coords, grid_size_axis = self.get_volume_coords(resolution=resolution)
            coords.to(self.DEVICE)
            coords_batches = torch.split(coords, coords.shape[0] // 1000)

            sdf = torch.tensor([]).to(self.DEVICE)

            local_generator = torch.Generator().manual_seed(int(time.time()))

            if cls_num is None:
                cls_nums = max(cls_dict.keys()) + 1
                cls_num = int(torch.randint(low=0, high=cls_nums, size=(1,), generator=local_generator))

            for coords_batch in tqdm(coords_batches, desc=f"Reconstructing in `{epoch}th` epoch", leave=False):
                cls = torch.tensor([cls_num] * coords_batch.shape[0], dtype=torch.long).to(self.DEVICE)
                pred = sdf_decoder(cls, coords_batch)

                if sum(sdf.shape) == 0:
                    sdf = pred
                else:
                    sdf = torch.vstack([sdf, pred])

            mesh = self.extract_mesh(grid_size_axis=grid_size_axis, sdf=sdf, normalize=normalize, map_z_to_y=map_z_to_y)

            if mesh is not None:
                mesh.export(obj_path.replace(".obj", f"_{epoch}_{cls_dict[cls_num]}.obj"))

        sdf_decoder.train()
