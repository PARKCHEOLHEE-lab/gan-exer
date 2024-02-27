import ray
import numpy as np
import multiprocessing

from typing import Tuple
from shapely.geometry import Polygon
from shapely import affinity
from lirGAN.data import utils

from debugvisualizer.debugvisualizer import Plotter  # noqa: F401


class LargestInscribedRectangle:
    def __init__(self):
        pass

    @staticmethod
    def _get_lir_indices(binary_grid_shaped_polygon: np.ndarray) -> Tuple[int, int]:
        """Get top left, bottom right indices of binary grid-shaped polygon

        Args:
            binary_grid_shaped_polygon (np.ndarray): 2d binary grid-shaped polygon consisting of 0s, 1s

        Returns:
            Tuple[int, int]: top left index, bottom right index
        """

        _, col_num = binary_grid_shaped_polygon.shape
        height = np.zeros((col_num + 1,), dtype=int)
        max_area = 0
        top_left = ()
        bottom_right = ()

        for ri, row in enumerate(binary_grid_shaped_polygon):
            height[:-1] = np.where(row == 1, height[:-1] + 1, 0)

            stack = [-1]
            for ci in range(col_num + 1):
                while height[stack[-1]] > height[ci]:
                    hi = stack.pop()
                    h = height[hi]
                    w = ci - stack[-1] - 1

                    area = h * w
                    if max_area < area:
                        max_area = area

                        top_left = (ri - h + 1, stack[-1] + 1)
                        bottom_right = (ri, ci - 1)

                stack.append(ci)

        return top_left, bottom_right

    @staticmethod
    @ray.remote
    def _get_each_lir_with_ray(
        rotation_degree: float, rotation_anchor: np.ndarray, binary_grid_shaped_polygon: np.ndarray
    ) -> Polygon:
        """Get the largest inscribed rectangle aligned by global xy axes on a given polygon
           and convert it to a vectorized polygon

        Args:
            rotation_degree (float): degree to rotate
            rotation_anchor (np.ndarray): anchor to rotate
            binary_grid_shaped_polygon (np.ndarray): 2d binary grid-shaped polygon consisting of 0s, 1s

        Returns:
            Polygon: the largest rectangle aligned by global xy axes on a given polygon
        """

        top_left, bottom_right = LargestInscribedRectangle._get_lir_indices(binary_grid_shaped_polygon)

        top_left_row, top_left_col = top_left
        bottom_right_row, bottom_right_col = bottom_right

        lir = np.zeros_like(binary_grid_shaped_polygon)
        lir[top_left_row : bottom_right_row + 1, top_left_col : bottom_right_col + 1] = 1

        lir_polygon = Polygon(utils.vectorize_polygon_from_array(lir))

        inverted_lir_polygon = affinity.rotate(geom=lir_polygon, angle=-rotation_degree, origin=rotation_anchor)

        return inverted_lir_polygon

    @staticmethod
    def _get_each_lir_without_ray(
        rotation_degree: float, rotation_anchor: np.ndarray, binary_grid_shaped_polygon: np.ndarray
    ) -> Polygon:
        """Get the largest inscribed rectangle aligned by global xy axes on a given polygon
           and convert it to a vectorized polygon

        Args:
            rotation_degree (float): degree to rotate
            rotation_anchor (np.ndarray): anchor to rotate
            binary_grid_shaped_polygon (np.ndarray): 2d binary grid-shaped polygon consisting of 0s, 1s

        Returns:
            Polygon: the largest rectangle aligned by global xy axes on a given polygon
        """

        top_left, bottom_right = LargestInscribedRectangle._get_lir_indices(binary_grid_shaped_polygon)

        top_left_row, top_left_col = top_left
        bottom_right_row, bottom_right_col = bottom_right

        lir = np.zeros_like(binary_grid_shaped_polygon)
        lir[top_left_row : bottom_right_row + 1, top_left_col : bottom_right_col + 1] = 1

        lir_polygon = Polygon(utils.vectorize_polygon_from_array(lir))

        inverted_lir_polygon = affinity.rotate(geom=lir_polygon, angle=-rotation_degree, origin=rotation_anchor)

        return inverted_lir_polygon

    @utils.runtime_calculator
    def _get_largest_inscribed_rectangle(
        self,
        coordinates: np.ndarray,
        canvas_size: np.ndarray,
        lir_rotation_degree_interval: float,
        use_ray: bool = True,
    ) -> Polygon:
        """Find the largest inscribed rectangle on a given polygon(coordinates)

        Args:
            coordinates (np.ndarray): polygon
            canvas_size (np.ndarray): binary grid size

        Returns:
            Polygon: the largest rectangle on a given polygon
        """

        lirs = []
        lir_args = []

        random_coordinates_polygon = Polygon(coordinates)
        rotation_anchor = random_coordinates_polygon.centroid

        rotation_degree = 0
        while rotation_degree < 360:
            rotated_random_polygon = affinity.rotate(
                geom=random_coordinates_polygon,
                angle=rotation_degree,
                origin=rotation_anchor,
            )

            binary_grid_shaped_polygon = utils.get_binary_grid_shaped_polygon(
                coordinates=np.array(rotated_random_polygon.boundary.coords).astype(np.int32),
                canvas_size=canvas_size,
            )

            lir_args.append(
                (
                    rotation_degree,
                    rotation_anchor.coords[0],
                    binary_grid_shaped_polygon,
                )
            )

            rotation_degree += lir_rotation_degree_interval

        if use_ray:
            if not ray.is_initialized():
                ray.init(num_cpus=multiprocessing.cpu_count())

            lir_futures = [LargestInscribedRectangle._get_each_lir_with_ray.remote(*args) for args in lir_args]
            lirs = ray.get(lir_futures)

        else:
            lirs = [LargestInscribedRectangle._get_each_lir_without_ray(*args) for args in lir_args]

        return max(lirs, key=lambda p: p.area)
