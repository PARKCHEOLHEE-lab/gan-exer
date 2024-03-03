import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def get_closed_mesh(_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Attempt to close an open mesh by filling holes.

    Args:
        mesh (trimesh.Trimesh): The open mesh to close.

    Returns:
        trimesh.Trimesh: The potentially closed mesh.
    """

    mesh = _mesh.copy()
    mesh.fill_holes()

    if not mesh.is_watertight:
        print(f"mesh.is_watertight: {mesh.is_watertight}")

    return mesh


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
        mesh = get_normalized_mesh(mesh)

    if map_y_to_z:
        mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

    mesh.path = path

    return mesh


def plot_mesh(mesh: trimesh.Trimesh) -> None:
    """Visualize the mesh using matplotlib

    Args:
        mesh (trimesh.Trimesh): The mesh to visualize.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract mesh vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Plot the mesh
    ax.add_collection3d(Poly3DCollection(vertices[faces], alpha=1, facecolor="white", linewidths=1, edgecolors="gray"))

    # Auto scale to the mesh size
    scale = np.concatenate([vertices.min(axis=0), vertices.max(axis=0)]).reshape(2, -1)
    mid = np.mean(scale, axis=0)
    max_range = (scale[1] - scale[0]).max() / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()


if __name__ == "__main__":
    plot_mesh(load_mesh(r"deepSDF\data\wave.obj", normalize=True, map_y_to_z=True))
