import random
import numpy as np
import matplotlib.pyplot as plt

import os

SEED = 777

class PointSampler:
    def __init__(self, output_size, seed=SEED):
        assert isinstance(output_size, int)
        self.output_size = output_size

        random.seed(seed)
        np.random.seed(seed)
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points


class Normalize:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud

    
def pcshow(*pcs, labels=None, axis_off=False, figsize=None):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    if labels is not None:
        assert len(pcs) == len(
            labels
        ), "The length between the given `args` and `labels` is different."

    if figsize is None:
        figsize = (6, 6)
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("plasma")
    num_args = len(pcs)
    colors = [cmap(i / num_args) for i in range(num_args)]

    for i, point_cloud in enumerate(pcs):
        x, y, z = point_cloud.T

        if labels is not None:
            ax.label(f"{labels[i]}")
            
        ax.scatter(x, y, z, c=colors[i], marker="o", s=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    if labels is not None:
        ax.legend()
    
    ax.set_aspect("equal")

    if axis_off:
        ax.axis("off")

    plt.show()