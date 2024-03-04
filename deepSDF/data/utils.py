import trimesh
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_mesh(mesh: trimesh.Trimesh, points: np.ndarray = None, only_points: bool = False) -> None:
    """
    Visualize the mesh using matplotlib, with an option to also plot sampled points or only points.

    Args:
        mesh (trimesh.Trimesh): The mesh to visualize.
        points (np.ndarray, optional): An array of points to plot along with the mesh. Defaults to None.
        only_points (bool, optional): If True, only points will be visualized. Defaults to False.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if not only_points:
        vertices = mesh.vertices
        faces = mesh.faces

        mesh_collection = Poly3DCollection(
            vertices[faces], alpha=0.8, facecolor="white", linewidths=0.5, edgecolors="gray"
        )
        ax.add_collection3d(mesh_collection)

        scale = np.concatenate([vertices.min(axis=0), vertices.max(axis=0)]).reshape(2, -1)
        mid = np.mean(scale, axis=0)
        max_range = (scale[1] - scale[0]).max() / 2
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="red", s=10)
        if only_points:
            min_vals = points.min(axis=0)
            max_vals = points.max(axis=0)
            max_range = (max_vals - min_vals).max() / 2
            mid = (max_vals + min_vals) / 2
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()
