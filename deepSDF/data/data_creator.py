import os
import trimesh
import numpy as np
import commonutils
import point_cloud_utils as pcu

from deepSDF.config import Configuration


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

        volume_points_sampled = np.random.rand(n_volume_sampling, 3) * 2 - 1

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
        self.raw_data_path = raw_data_path
        self.save_path = save_path
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

        fi = 0
        for file in os.listdir(self.raw_data_path):
            if not file.endswith(".obj"):
                continue

            path = os.path.join(self.raw_data_path, file)

            mesh = self.load_mesh(path, normalize=True, map_z_to_y=True)

            xyz = self.sample_pts(mesh, self.n_surface_sampling, self.n_bbox_sampling, self.n_volume_sampling)
            sdf, *_ = pcu.signed_distance_to_mesh(xyz, mesh.vertices, mesh.faces)
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
        is_debug_mode=False,
    )

    data_creator.create()
