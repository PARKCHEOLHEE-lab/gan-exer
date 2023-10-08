import trimesh
import numpy as np
import os

import binvox_rw
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
    
def plot_binvox(path: str, map_y_to_z: bool = False) -> None:

    with open(path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)

    data = model.data

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = data.nonzero()
    if map_y_to_z:
        y, z = z, y

    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    plt.show()