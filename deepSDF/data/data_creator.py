import os
import ray
import trimesh
import numpy as np
import commonutils
import multiprocessing
import point_cloud_utils as pcu

from tqdm import tqdm
from typing import List, Union
from deepSDF.config import Configuration


class DataCreatorHelper:
    MIN_BOUND = "min_bound"
    CENTER = "center"
    CENTER_WITHOUT_Z = "center_without_z"

    @staticmethod
    @ray.remote
    def calculate_max_length(
        paths: List[str],
        map_z_to_y: bool = False,
        check_watertight: bool = True,
        translate_mode: Union[MIN_BOUND, CENTER, CENTER_WITHOUT_Z] = CENTER_WITHOUT_Z,
        save_html: bool = False,
    ) -> float:
        """
        Calculate the maximum length of the given mesh.

        Args:
            paths (List[str]): Paths of the meshes to calculate the maximum length.

        Returns:
            float: The maximum length of the given mesh
        """

        if save_html:
            commonutils.add_debugvisualizer(globals())

        meshes = []
        max_length = 0

        for path in paths:
            mesh = DataCreatorHelper.load_mesh(
                path,
                normalize=False,
                map_z_to_y=map_z_to_y,
                check_watertight=check_watertight,
                translate_mode=translate_mode,
            )

            if not mesh.is_watertight:
                print(f"{path} is not watertight")
                continue

            length = np.max(np.linalg.norm(mesh.vertices, axis=1))
            if length > max_length:
                max_length = length

            meshes.append(mesh)

            if save_html:
                save_name = os.path.basename(path).replace(".obj", ".html")
                print(f"saving: {save_name}")

                globals()["Plotter"](
                    mesh,
                    globals()["geometry"].Point(mesh.vertices[np.argmax(np.linalg.norm(mesh.vertices, axis=1))]),
                    globals()["geometry"].Point(0, 0),
                    map_z_to_y=False,
                ).save(save_name)

        return meshes, max_length

    @staticmethod
    def get_normalized_mesh(_mesh: trimesh.Trimesh, max_length: float = None) -> trimesh.Trimesh:
        """Normalize to 0 ~ 1 values the given mesh

        Args:
            mesh (trimesh.Trimesh): Given mesh to normalize

        Returns:
            trimesh.Trimesh: Normalized mesh
        """

        mesh = _mesh.copy()

        if max_length is not None:
            length = max_length
        else:
            length = np.max(np.linalg.norm(mesh.vertices, axis=1))

        mesh.vertices = mesh.vertices * (1.0 / length)

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
        translate_mode: Union[MIN_BOUND, CENTER, CENTER_WITHOUT_Z] = CENTER_WITHOUT_Z,
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

        if check_watertight and not mesh.is_watertight:
            vertices, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=100000)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if map_z_to_y:
            mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

        if translate_mode == DataCreatorHelper.MIN_BOUND:
            vector = mesh.bounds[0]
        elif translate_mode == DataCreatorHelper.CENTER:
            vector = np.mean(mesh.vertices, axis=0)
        elif translate_mode == DataCreatorHelper.CENTER_WITHOUT_Z:
            vector = mesh.bounds.sum(axis=0) * 0.5
            vector[2] = mesh.bounds[0][2]
        else:
            raise ValueError(f"Invalid translate mode: {translate_mode}")

        mesh.vertices -= vector

        if normalize:
            mesh = DataCreatorHelper.get_normalized_mesh(mesh, max_length=max_length)

        mesh.path = path

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
        dynamic_sampling: bool,
        is_debug_mode: bool = False,
    ) -> None:
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.translate_mode = translate_mode
        self.dynamic_sampling = dynamic_sampling
        self.is_debug_mode = is_debug_mode

        self.n_surface_sampling = n_surface_sampling
        self.n_bbox_sampling = n_bbox_sampling
        self.n_volume_sampling = n_volume_sampling

        if self.is_debug_mode:
            commonutils.add_debugvisualizer(globals())

    @commonutils.runtime_calculator
    def create(self) -> None:
        """Create data for training sdf decoder"""

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        paths = [
            os.path.join(self.raw_data_path, file) for file in os.listdir(self.raw_data_path) if file.endswith(".obj")
        ]

        ray.init(num_cpus=multiprocessing.cpu_count())

        futures = self.calculate_max_length.remote(
            paths, map_z_to_y=True, check_watertight=True, translate_mode=self.translate_mode, save_html=False
        )

        meshes, max_length = ray.get(futures)

        cls = 0
        for mesh in tqdm(meshes, desc="Preprocessing"):
            normalized_mesh = self.get_normalized_mesh(mesh, max_length=max_length)

            centralized_mesh = normalized_mesh.copy()
            centralized_mesh.vertices += np.array([0.5, 0.5, 0])

            if self.dynamic_sampling:
                (
                    self.n_surface_sampling,
                    self.n_bbox_sampling,
                    self.n_volume_sampling,
                ) = Configuration.get_dynamic_sampling_size(mesh_vertices_count=mesh.vertices.shape[0])

            print(
                f"mesh_vertices_count: {mesh.vertices.shape[0]}",
                f"n_total_sampling: {self.n_surface_sampling + self.n_bbox_sampling + self.n_volume_sampling}",
            )

            xyz = self.sample_pts(
                centralized_mesh, self.n_surface_sampling, self.n_bbox_sampling, self.n_volume_sampling
            )

            sdf, *_ = pcu.signed_distance_to_mesh(xyz, centralized_mesh.vertices, centralized_mesh.faces)
            sdf = np.expand_dims(sdf, axis=1)

            cls_name = os.path.basename(mesh.path).split(".")[0]

            np.savez(
                os.path.join(self.save_path, f"{cls_name}.npz"),
                xyz=xyz,
                sdf=sdf,
                cls=cls,
                cls_name=cls_name,
            )

            cls += 1


if __name__ == "__main__":
    data_creator = DataCreator(
        n_surface_sampling=Configuration.N_SURFACE_SAMPLING,
        n_bbox_sampling=Configuration.N_BBOX_SAMPLING,
        n_volume_sampling=Configuration.N_VOLUME_SAMPLING,
        raw_data_path=Configuration.RAW_DATA_PATH,
        save_path=Configuration.SAVE_DATA_PATH_DYNAMIC_SAMPLED,
        translate_mode=DataCreatorHelper.CENTER_WITHOUT_Z,
        dynamic_sampling=True,
        is_debug_mode=False,
    )

    data_creator.create()
