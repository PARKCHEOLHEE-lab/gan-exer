import trimesh
import numpy as np
import os

import scipy
import binvox_rw
import matplotlib.pyplot as plt
import shutil
import copy

from typing import List, Tuple

class PreprocessConfig:    
    OBJ_FORMAT = ".obj"
    BINVOX_FORMAT = ".binvox"
    
    BINVOX_RESOLUTION = 36
    ROTATION_MAX = 360
    ROTATION_DIVIDER = 10
    ROTATION_STEP = ROTATION_MAX / ROTATION_DIVIDER

    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)

    DATA_BASE_DIR = "data"
    DATA_ORIGINAL_DIR = "original"
    DATA_PREPROCESSED_DIR = "preprocessed"

    DATA_ORIGINAL_DIR_MERGED = os.path.join(DATA_BASE_DIR, DATA_ORIGINAL_DIR)
    DATA_PREPROCESSED_DIR_MERGED = os.path.join(DATA_BASE_DIR, DATA_PREPROCESSED_DIR)


class Utils:
    @staticmethod
    def normalize_mesh(mesh: trimesh.Trimesh) -> None:
        """Normalize to 0 ~ 1 values the given mesh

        Args:
            mesh (trimesh.Trimesh): Given mesh to normalize
        """

        verts = mesh.vertices
        centers = np.mean(verts, axis=0)
        verts = verts - centers
        length = np.max(np.linalg.norm(verts, 2, axis=1))
        verts = verts * (1. / length)
        
        mesh.vertices = verts
    
    @staticmethod
    def get_rotated_mesh(mesh: trimesh.Trimesh, angle: float, axis: Tuple[int] = PreprocessConfig.Z) -> trimesh.Trimesh:
        """Return the rotated mesh by the given mesh and angle

        Args:
            mesh (trimesh.Trimesh): Given mesh
            angle (float): Angle to rotate
            axis (Tuple[int], optional): Axis to rotate. Defaults to PreprocessConfig.Z.

        Returns:
            trimesh.Trimesh: Rotated mesh
        """

        rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
        rotated_mesh = copy.deepcopy(mesh)
        rotated_mesh.apply_transform(rotation_matrix)
        
        return rotated_mesh
        
    @staticmethod
    def load_mesh(path: str, normalize: bool = False, map_y_to_z: bool = False) -> trimesh.Trimesh:
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
            Utils.normalize_mesh(mesh)
            
        if map_y_to_z:
            mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

        return mesh

    @staticmethod
    def mesh_to_binvox(
        mesh: trimesh.Trimesh,
        save_path: str,
        resolution: int, 
    ) -> None:
        """Convert mesh that is shaped .obj format to .binvox

        Args:
            mesh (trimesh.Trimesh): Given mesh
            save_path (str): Path to save
            resolution (int): Binary voxel grid resolution
        """

        data_name = mesh.path.split("\\")[-1]
        merged_save_path = os.path.join(save_path, data_name)
        mesh.export(merged_save_path)
                    
        command = f"binvox -cb -e -d {resolution} {merged_save_path}"
        os.system(command)
        
        os.remove(os.path.join(save_path, "material.mtl"))
        os.remove(os.path.join(save_path, "material_0.png"))
        os.remove(os.path.join(save_path, data_name))
    
    @staticmethod
    def plot_binvox(
        data_list: List[np.ndarray], 
        plot_voxels: bool = False, 
        downsample_rate: float = 1.0,
        title: str = ""
    ) -> None:
        """Plot .binvox data

        Args:
            data_list (List[np.ndarray]): Given binvox data list
            plot_voxels (bool, optional): Whether plotting voxel or scatter. Defaults to False.
            downsample_rate (float, optional): Resolution for only plotting. Defaults to 1.0.
            title (str, optional): Title to plot. Defaults to "".
        """

        data_list_length = len(data_list)
        figure = plt.figure(figsize=(data_list_length * 2, data_list_length / 3))
        figure.suptitle(title)

        color = "blue"

        for n in range(1, data_list_length + 1):
            ax = figure.add_subplot(1, data_list_length, n, projection="3d")
            
            data = data_list[n - 1]
            
            x, y, z = data.nonzero()

            if plot_voxels:
                data = scipy.ndimage.zoom(data, downsample_rate, order=0)
                ax.voxels(data, facecolors=color)
            else:
                ax.scatter(x, y, z, c=color)

        ax.set_aspect("equal")
        figure.tight_layout()

        plt.show()

class Preprocessor(Utils):
    def preprocess(self, overwrite=True) -> None:
        """Main function for preprocessing data

        Args:
            overwrite (bool, optional): whether data overwriting. Defaults to True.
        """
        
        for data_name in os.listdir(PreprocessConfig.DATA_ORIGINAL_DIR_MERGED):
            
            each_save_dir = os.path.join(PreprocessConfig.DATA_PREPROCESSED_DIR_MERGED, data_name)
            if not os.path.isdir(each_save_dir):
                os.mkdir(each_save_dir)
            
            each_obj_data_path = os.path.join(
                PreprocessConfig.DATA_ORIGINAL_DIR_MERGED,
                data_name, 
                data_name + PreprocessConfig.OBJ_FORMAT
            )
            
            mesh = self.load_mesh(path=each_obj_data_path, normalize=True, map_y_to_z=True)

            if overwrite:
                for file_name in os.listdir(each_save_dir):
                    if file_name.endswith(PreprocessConfig.BINVOX_FORMAT):
                        os.remove(os.path.join(each_save_dir, file_name))

            rotation_degree = 0
            binvox_number = 0
            
            data_list: List[np.ndarray]
            data_list = []

            while rotation_degree < PreprocessConfig.ROTATION_MAX:
                rotated_mesh = self.get_rotated_mesh(mesh=mesh, angle=np.radians(rotation_degree))
                rotated_mesh.path = each_obj_data_path
                
                rotation_degree += PreprocessConfig.ROTATION_STEP

                self.mesh_to_binvox(
                    mesh=rotated_mesh,
                    save_path=each_save_dir, 
                    resolution=PreprocessConfig.BINVOX_RESOLUTION, 
                )

                binvox_number_string = "_" + str(binvox_number) if binvox_number > 0 else ""
            
                each_binvox_data_path = os.path.join(
                    each_save_dir, data_name + binvox_number_string + PreprocessConfig.BINVOX_FORMAT
                )
                
                binvox_number += 1
                
                with open(each_binvox_data_path, 'rb') as f:
                    model = binvox_rw.read_as_3d_array(f)
                    
                    data_list.append(model.data)
                    
            self.plot_binvox(data_list=data_list, plot_voxels=True, title=data_name)