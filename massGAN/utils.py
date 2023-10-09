import trimesh
import numpy as np
import os

import scipy
import matplotlib.pyplot as plt


def mesh2binvox(path: str, resolution: int, normalize: bool = True, overwrite: bool = True) -> None:

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
    
    if plot_voxels:
        data = scipy.ndimage.zoom(data, downsample_rate, order=0)
        ax.voxels(data, facecolors=color)
    else:
        x, y, z = data.nonzero()
        ax.scatter(x, y, z, c=color)

    ax.set_aspect("equal")
    
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    
    ax.set_title(title)
    

    plt.show()