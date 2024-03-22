import os
import ray
import trimesh
import numpy as np
import commonutils
import point_cloud_utils as pcu

from tqdm import tqdm
from typing import List, Union
from deepSDF.config import Configuration


class DataCreatorHelper:
    MIN_BOUND = "min_bound"
    CENTER = "center"

    @staticmethod
    @ray.remote
    def calculate_max_length(paths: List[str], translate_mode: Union[MIN_BOUND, CENTER] = CENTER) -> float:
        """
        Calculate the maximum length of the given mesh.

        Args:
            paths (List[str]): Paths of the meshes to calculate the maximum length.

        Returns:
            float: The maximum length of the given mesh
        """

        max_length = 0

        for path in paths:
            mesh = trimesh.load(path)

            if isinstance(mesh, trimesh.Scene):
                geo_list = []
                for g in mesh.geometry.values():
                    geo_list.append(g)
                mesh = trimesh.util.concatenate(geo_list)

            mesh.fix_normals(multibody=True)

            verts = mesh.vertices

            if translate_mode == DataCreatorHelper.MIN_BOUND:
                vector = mesh.bounds[0]
            elif translate_mode == DataCreatorHelper.CENTER:
                vector = np.mean(verts, axis=0)
            else:
                raise ValueError(f"Invalid translate mode: {translate_mode}")

            verts = verts - vector
            length = np.max(np.linalg.norm(verts, axis=1))
            if length > max_length:
                max_length = length

        return max_length

    @staticmethod
    def get_normalized_mesh(
        _mesh: trimesh.Trimesh, max_length: float = None, translate_mode: Union[MIN_BOUND, CENTER] = CENTER
    ) -> trimesh.Trimesh:
        """Normalize to 0 ~ 1 values the given mesh

        Args:
            mesh (trimesh.Trimesh): Given mesh to normalize

        Returns:
            trimesh.Trimesh: Normalized mesh
        """

        mesh = _mesh.copy()

        verts = mesh.vertices

        if translate_mode == DataCreatorHelper.MIN_BOUND:
            vector = mesh.bounds[0]
        elif translate_mode == DataCreatorHelper.CENTER:
            vector = np.mean(verts, axis=0)
        else:
            raise ValueError(f"Invalid translate mode: {translate_mode}")

        verts = verts - vector

        if max_length is not None:
            length = max_length
        else:
            length = np.max(np.linalg.norm(verts, axis=1))

        verts = verts * (1.0 / length)

        mesh.vertices = verts

        return mesh

    @staticmethod
    def get_closed_mesh(_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Attempt to close an open mesh by filling holes.

        Args:
            mesh (trimesh.Trimesh): The open mesh to close.

        Returns:
            trimesh.Trimesh: The potentially closed mesh.
        """

        mesh = _mesh.copy()
        mesh.fill_holes()

        return mesh

    @staticmethod
    @commonutils.runtime_calculator
    def sample_pts(
        mesh: trimesh.Trimesh,
        n_surface_sampling: int,
        n_bbox_sampling: int,
        n_volume_sampling: int,
        sigma: float = 0.01,
        with_surface_points_noise: bool = True,
    ) -> np.ndarray:
        """
        Sample a given number of points uniformly from the surface of a mesh.

        Args:
            mesh (trimesh.Trimesh): The mesh from which to sample points.
            n_surface_sampling (int): The number of points to sample from the surface.
            n_bbox_sampling (int): The number of points to sample from the bounding box.
            n_volume_sampling (int): The number of points to sample from the volume.

        Returns:
            np.ndarray: An array of sampled points (shape: [num_samples, 3]).
        """

        if not with_surface_points_noise:
            sigma = 0

        surface_points_sampled, _ = trimesh.sample.sample_surface(mesh, n_surface_sampling)
        surface_points_sampled += np.random.normal(0, sigma, surface_points_sampled.shape)

        bbox_points_sampled = np.random.uniform(low=mesh.bounds[0], high=mesh.bounds[1], size=[n_bbox_sampling, 3])

        # volume_points_sampled = np.random.rand(n_volume_sampling, 3) * 2 - 1
        volume_points_sampled = np.random.rand(n_volume_sampling, 3)

        xyz = np.concatenate([surface_points_sampled, bbox_points_sampled, volume_points_sampled], axis=0)

        return xyz

    @staticmethod
    def load_mesh(
        path: str,
        normalize: bool = False,
        map_z_to_y: bool = False,
        check_watertight: bool = True,
        max_length: float = None,
        translate_mode: Union[MIN_BOUND, CENTER] = CENTER,
    ) -> trimesh.Trimesh:
        """Load mesh data from .obj file

        Args:
            path (str): Path to load
            normalize (bool, optional): Whether normalizing mesh. Defaults to False.
            map_y_to_z (bool, optional): Change axes (y to z, z to y). Defaults to False.

        Returns:
            trimesh.Trimesh: Loaded mesh
        """

        mesh = trimesh.load(path)

        if isinstance(mesh, trimesh.Scene):
            geo_list = []
            for g in mesh.geometry.values():
                geo_list.append(g)
            mesh = trimesh.util.concatenate(geo_list)

        mesh.fix_normals(multibody=True)

        if normalize:
            mesh = DataCreatorHelper.get_normalized_mesh(mesh, max_length=max_length, translate_mode=translate_mode)

        if map_z_to_y:
            mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

        mesh.path = path

        if check_watertight and not mesh.is_watertight:
            vertices, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=100000)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return mesh


class DataCreator(DataCreatorHelper):
    def __init__(
        self,
        n_surface_sampling: int,
        n_bbox_sampling: int,
        n_volume_sampling: int,
        raw_data_path: str,
        save_path: str,
        translate_mode: str,
        is_debug_mode: bool = False,
    ) -> None:
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.translate_mode = translate_mode
        self.is_debug_mode = is_debug_mode

        self.n_surface_sampling = n_surface_sampling
        self.n_bbox_sampling = n_bbox_sampling
        self.n_volume_sampling = n_volume_sampling

        if self.is_debug_mode:
            commonutils.add_debugvisualizer(globals())

    def create(self) -> None:
        """_summary_"""

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        paths = [
            os.path.join(Configuration.RAW_DATA_PATH, file)
            for file in os.listdir(Configuration.RAW_DATA_PATH)
            if file.endswith(".obj")
        ]

        ray.init()
        max_length = DataCreatorHelper.calculate_max_length.remote(paths, translate_mode=self.translate_mode)
        max_length = ray.get(max_length)

        fi = 0
        for path in tqdm(paths, desc="Preprocessing"):
            mesh = self.load_mesh(
                path,
                normalize=True,
                map_z_to_y=True,
                check_watertight=True,
                max_length=max_length,
                translate_mode=self.translate_mode,
            )

            if not mesh.is_watertight:
                print(f"{path} is not watertight")
                continue

            mesh_central = mesh.copy()
            mesh_central.vertices += np.array([0.5, 0.5, 0]) - (mesh.bounds.sum(axis=0) * 0.5) * np.array([1, 1, 0])

            xyz = self.sample_pts(mesh_central, self.n_surface_sampling, self.n_bbox_sampling, self.n_volume_sampling)
            sdf, *_ = pcu.signed_distance_to_mesh(xyz, mesh_central.vertices, mesh_central.faces)
            sdf = np.expand_dims(sdf, axis=1)

            np.savez(os.path.join(self.save_path, str(fi) + ".npz"), xyz=xyz, sdf=sdf, cls=fi)

            fi += 1


if __name__ == "__main__":
    data_creator = DataCreator(
        n_surface_sampling=Configuration.N_SURFACE_SAMPLING,
        n_bbox_sampling=Configuration.N_BBOX_SAMPLING,
        n_volume_sampling=Configuration.N_VOLUME_SAMPLING,
        raw_data_path=Configuration.RAW_DATA_PATH,
        save_path=Configuration.SAVE_DATA_PATH,
        translate_mode=DataCreatorHelper.MIN_BOUND,
        is_debug_mode=True,
    )

    data_creator.create()
