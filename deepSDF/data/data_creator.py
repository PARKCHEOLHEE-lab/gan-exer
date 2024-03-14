import os
import ray
import trimesh
import numpy as np
import commonutils
import point_cloud_utils as pcu

from typing import Tuple


class DataCreatorHelper:
    @staticmethod
    def get_normalized_mesh(_mesh: trimesh.Trimesh) -> None:
        """Normalize to 0 ~ 1 values the given mesh

        Args:
            mesh (trimesh.Trimesh): Given mesh to normalize
        """

        mesh = _mesh.copy()

        verts = mesh.vertices
        centers = np.mean(verts, axis=0)
        verts = verts - centers
        length = np.max(np.linalg.norm(verts, 2, axis=1))
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

        surface_points_sampled, _ = trimesh.sample.sample_surface(mesh, n_surface_sampling)
        surface_points_sampled += np.random.normal(0, sigma, surface_points_sampled.shape)

        bbox_points_sampled = np.random.uniform(low=mesh.bounds[0], high=mesh.bounds[1], size=[n_bbox_sampling, 3])

        volume_points_sampled = np.random.rand(n_volume_sampling, 3) * max(mesh.bounds[1] - mesh.bounds[0])
        volume_points_sampled -= np.mean(volume_points_sampled, axis=0)
        volume_points_sampled += np.mean(mesh.vertices, axis=0)

        xyz = np.concatenate([surface_points_sampled, bbox_points_sampled, volume_points_sampled], axis=0)

        return xyz

    @staticmethod
    def load_mesh(
        path: str, normalize: bool = False, map_z_to_y: bool = False, check_watertight: bool = True
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
            mesh = DataCreatorHelper.get_normalized_mesh(mesh)

        if map_z_to_y:
            mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

        mesh.path = path

        if check_watertight and not mesh.is_watertight:
            vertices, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=100000)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return mesh

    @staticmethod
    @ray.remote
    def _is_visibile_face(
        mesh: trimesh.Trimesh, raycasting_origins: trimesh.caching.TrackedArray, face_index: int
    ) -> Tuple[int, bool]:
        """_summary_

        Args:
            mesh (trimesh.Trimesh): _description_
            ray_origins (trimesh.caching.TrackedArray): _description_
            face_index (int): _description_

        Returns:
            Tuple[int, bool]: _description_
        """

        face = mesh.triangles[face_index]
        hit_count = 0

        for point in face:
            raycasting_directions = -(raycasting_origins - point)
            faces_hit = mesh.ray.intersects_first(raycasting_origins, raycasting_directions)

            if face_index in faces_hit:
                hit_count += 1

            if hit_count > 0:
                break

        return face_index, hit_count > 0

    @staticmethod
    @ray.remote
    def compute_sdf_batch(noisy_points_batch, mesh_vertices, mesh_faces):
        mesh_vertices_np = np.array(mesh_vertices).astype(np.float32)  # or np.float64 if higher precision is needed
        mesh_faces_np = np.array(mesh_faces).astype(np.int32)  # or np.int64 if needed
        sdf, _, _ = pcu.signed_distance_to_mesh(noisy_points_batch, mesh_vertices_np, mesh_faces_np)

        return sdf

    @staticmethod
    @commonutils.runtime_calculator
    def compute_sdf(
        mesh: trimesh.Trimesh, points: np.ndarray, sigma: float = 0.01, with_noise: bool = True
    ) -> np.ndarray:
        """
        Compute the Signed Distance Function (SDF) for a set of points given a mesh.

        Args:
            mesh (trimesh.Trimesh): The mesh for which to compute the SDF.
            points (np.ndarray): An array of points (shape: [N, 3]) for which to compute the SDF values.

        Returns:
            np.ndarray: An array of SDF values for the given points.
        """

        if not with_noise:
            sigma = 0

        noise = np.random.normal(0, sigma, points.shape)
        noisy_points = points + noise

        sdf, *_ = pcu.signed_distance_to_mesh(noisy_points, mesh.vertices, mesh.faces)

        return sdf


class DataCreator(DataCreatorHelper):
    def __init__(
        self,
        n_surface_sampling: int,
        n_bbox_sampling: int,
        n_volume_sampling: int,
        raw_data_path: str,
        save_path: str,
        is_debug_mode: bool = False,
    ) -> None:
        self.n_surface_sampling = n_surface_sampling
        self.n_bbox_sampling = n_bbox_sampling
        self.n_volume_sampling = n_volume_sampling
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.is_debug_mode = is_debug_mode

        if self.is_debug_mode:
            from debugvisualizer.debugvisualizer import Plotter
            from shapely import geometry

            globals()["Plotter"] = Plotter
            globals()["geometry"] = geometry

    def create(self) -> None:
        """_summary_"""

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        ray.init()

        fi = 0
        for file in os.listdir(self.raw_data_path):
            if not file.endswith(".obj"):
                continue

            path = os.path.join(self.raw_data_path, file)

            mesh = self.load_mesh(path, normalize=True, map_z_to_y=True)

            xyz = self.sample_pts(mesh, self.n_surface_sampling, self.n_bbox_sampling, self.n_volume_sampling)
            sdf = np.expand_dims(self.compute_sdf(mesh, xyz), axis=1)

            np.savez(os.path.join(self.save_path, str(fi) + ".npz"), xyz=xyz, sdf=sdf)

            fi += 1


if __name__ == "__main__":
    data_creator = DataCreator(
        n_surface_sampling=10000,
        n_bbox_sampling=10000,
        n_volume_sampling=3000,
        raw_data_path="deepSDF/data/raw",
        save_path="deepSDF/data/preprocessed",
        is_debug_mode=True,
    )

    data_creator.create()
