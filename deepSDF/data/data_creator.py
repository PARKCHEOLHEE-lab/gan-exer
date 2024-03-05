import os
import ray
import trimesh
import numpy as np
import commonutils

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
    def sample_surface_points(
        mesh: trimesh.Trimesh, n_surface_sampling: int, mesh_visible_faces: dict = None
    ) -> np.ndarray:
        """
        Sample a given number of points uniformly from the surface of a mesh.

        Args:
            mesh (trimesh.Trimesh): The mesh from which to sample points.
            num_samples (int): The number of points to sample.

        Returns:
            np.ndarray: An array of sampled points (shape: [num_samples, 3]).
        """

        points, face_indices = trimesh.sample.sample_surface(mesh, n_surface_sampling)

        if mesh_visible_faces is not None:
            filtered_points = []
            filtered_indices = []

            for point, face_index in zip(points, face_indices):
                if mesh_visible_faces[face_index] == 1:
                    filtered_points.append(point)
                    filtered_indices.append(face_index)

            points = np.array(filtered_points)

        return points

    @staticmethod
    def load_mesh(path: str, normalize: bool = False, map_z_to_y: bool = False) -> trimesh.Trimesh:
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
    @commonutils.runtime_calculator
    def compute_visible_faces(mesh: trimesh.Trimesh, n_sphere_sampling: int, sphere_scaler: float = 1.5) -> dict:
        """_summary_

        Args:
            mesh (trimesh.Trimesh): _description_
            n_sphere_sampling (int): _description_
            sphere_scaler (float, optional): _description_. Defaults to 1.5.

        Returns:
            dict: _description_
        """

        mesh_visible_faces = {i: 0 for i in range(len(mesh.faces))}

        raycasting_origins = DataCreatorHelper.sample_surface_points(mesh.bounding_sphere, n_sphere_sampling)
        raycasting_origins *= sphere_scaler

        visibilities = [
            DataCreatorHelper._is_visibile_face.remote(mesh, raycasting_origins, face_index)
            for face_index in range(len(mesh.triangles))
        ]

        for visibility in ray.get(visibilities):
            face_index, is_visible = visibility
            if is_visible:
                mesh_visible_faces[face_index] = 1

        return mesh_visible_faces

    @staticmethod
    @commonutils.runtime_calculator
    def compute_sdf(mesh: trimesh.Trimesh, points: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """
        Compute the Signed Distance Function (SDF) for a set of points given a mesh.

        Args:
            mesh (trimesh.Trimesh): The mesh for which to compute the SDF.
            points (np.ndarray): An array of points (shape: [N, 3]) for which to compute the SDF values.

        Returns:
            np.ndarray: An array of SDF values for the given points.
        """
        if not mesh.is_watertight:
            mesh = DataCreatorHelper.get_closed_mesh(mesh)

        noise = np.random.normal(0, sigma, points.shape)
        noisy_points = points + noise

        return mesh.nearest.signed_distance(noisy_points)


class DataCreator(DataCreatorHelper):
    def __init__(
        self, n_surface_sampling: int, n_sphere_sampling: int, raw_data_path: str, is_debug_mode: bool = False
    ) -> None:
        self.n_surface_sampling = n_surface_sampling
        self.n_sphere_sampling = n_sphere_sampling
        self.raw_data_path = raw_data_path
        self.is_debug_mode = is_debug_mode

        if self.is_debug_mode:
            from debugvisualizer.debugvisualizer import Plotter
            from shapely import geometry

            globals()["Plotter"] = Plotter
            globals()["geometry"] = geometry

    def create(self) -> None:
        """_summary_"""

        ray.init()

        sdf_list = []
        xyz_list = []

        for file in os.listdir(self.raw_data_path):
            if not file.endswith(".obj"):
                continue

            path = os.path.join(self.raw_data_path, file)

            mesh = self.load_mesh(path, normalize=True, map_z_to_y=True)
            mesh_visible_faces = self.compute_visible_faces(mesh, self.n_sphere_sampling)

            xyz = self.sample_surface_points(mesh, self.n_surface_sampling, mesh_visible_faces=mesh_visible_faces)
            sdf = self.compute_sdf(mesh, xyz)

            xyz_list.append(xyz[np.where(sdf != np.nan)])
            sdf_list.append(sdf[np.where(sdf != np.nan)])


if __name__ == "__main__":
    data_creator = DataCreator(
        n_surface_sampling=250000,
        n_sphere_sampling=1000,
        raw_data_path="deepSDF/data/raw",
        is_debug_mode=True,
    )
    data_creator.create()

    # mesh = DataCreatorHelper.load_mesh(r"deepSDF\data\raw\wave.obj", normalize=True, map_y_to_z=True)
    # mesh_surface_sampled_points, _ = DataCreatorHelper.get_sampled_points_from_mesh(mesh, 100)
    # mesh_sphere_sampled_points, _ = DataCreatorHelper.get_sampled_points_from_mesh(mesh.bounding_sphere, 1000)

    # # sdfs = compute_sdf(mesh, sampled_points)

    # # print(sdfs)

    # plot_mesh(mesh, points=mesh_sphere_sampled_points, only_points=False)

    # # verts, faces, _, _ = measure.marching_cubes(sdfs, level=0)
