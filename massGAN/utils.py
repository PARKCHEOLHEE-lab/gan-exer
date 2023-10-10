import trimesh
import numpy as np
import os

import scipy
import matplotlib.pyplot as plt

class Utils:

    @staticmethod
    def mesh_to_binvox(path: str, resolution: int, normalize: bool = True, overwrite: bool = True) -> None:

        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            geo_list = []
            for g in mesh.geometry.values():
                geo_list.append(g)
            mesh = trimesh.util.concatenate(geo_list)

        mesh.fix_normals(multibody=True)

        if normalize:
            verts = mesh.vertices
            centers = np.mean(verts, axis=0)
            verts = verts - centers
            length = np.max(np.linalg.norm(verts, 2, axis=1))
            verts = verts * (1. / length)
            
            mesh.vertices = verts

        norm_path = path.replace(".obj", "_norm.obj")
        mesh.export(norm_path)
        
        if overwrite:
            binvox_path = f'{norm_path.replace("obj", "binvox")}'
            if os.path.exists(binvox_path):
                os.remove(binvox_path)

        command = f"binvox -cb -e -d {resolution} {norm_path}"
        os.system(command)
    
    @staticmethod
    def plot_binvox(
        data: np.ndarray, 
        map_y_to_z: bool = False, 
        plot_voxels: bool = False, 
        downsample_rate: float = 1.0,
        title: str = ""
    ) -> None:

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        
        if map_y_to_z:
            data = np.transpose(data, (0, 2, 1))

        color = "blue"
        
        x, y, z = data.nonzero()
        
        if plot_voxels:
            data = scipy.ndimage.zoom(data, downsample_rate, order=0)
            ax.voxels(data, facecolors=color)
        else:
            ax.scatter(x, y, z, c=color)

        ax.set_aspect("equal")
        
        ax.set_title(title)
        

        plt.show()
        
    @staticmethod
    def rotate_binvox(data: np.ndarray, degree: float, map_y_to_z: bool = False) -> np.ndarray:
        theta = np.radians(degree)
        matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ]
        )

        rotated_binvox = np.zeros_like(data)
        
        if map_y_to_z:
            data = np.transpose(data, (0, 2, 1))
        
        center = np.array([data.shape[0] / 2, data.shape[1] / 2, data.shape[2] / 2])

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    if data[x, y, z]:
                        coord = np.array([x, y, z]) - center
                        rotated_coord = np.dot(matrix, coord)
                        
                        rotated_coord += center
                        
                        rotated_x, rotated_y, rotated_z = np.round(rotated_coord).astype(int)

                        if (0 <= rotated_x < data.shape[0] and
                            0 <= rotated_y < data.shape[1] and
                            0 <= rotated_z < data.shape[2]):
                            rotated_binvox[rotated_x, rotated_y, rotated_z] = 1

        return rotated_binvox