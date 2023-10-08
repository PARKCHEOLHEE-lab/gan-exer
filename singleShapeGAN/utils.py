import binvox
import trimesh
import binvox_rw
import numpy as np
import os


def mesh2binvox(path: str, resolution: int, normalize: bool = True) -> None:

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

    command = f"binvox -cb -e -d {resolution} {norm_path}"
    os.system(command)