import numpy as np
import cv2

from shapely.geometry import Polygon
from shapely import affinity
from lirGAN.data import utils
import multiprocessing

import ray
ray.init(num_cpus=multiprocessing.cpu_count())

from debugvisualizer.debugvisualizer import Plotter


@ray.remote
def _get_lir_indices(binary_grid_shaped_lir: np.ndarray) -> np.ndarray:
    _, col_num = binary_grid_shaped_lir.shape
    height = np.zeros((col_num + 1,), dtype=int)
    max_area = 0
    top_left = ()
    bottom_right = ()

    for ri, row in enumerate(binary_grid_shaped_lir):
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
    
@ray.remote
def _get_each_lir(rotation_degree, rotation_anchor, binary_grid_shaped_polygon):
    # Call the remote function
    lir_indices_future = _get_lir_indices.remote(binary_grid_shaped_polygon)

    # Use ray.get() to retrieve the actual results and then unpack
    top_left, bottom_right = ray.get(lir_indices_future)
    
    top_left_row, top_left_col = top_left
    bottom_right_row, bottom_right_col = bottom_right
    
    lir = np.zeros_like(binary_grid_shaped_polygon)
    lir[top_left_row:bottom_right_row + 1, top_left_col:bottom_right_col + 1] = 1
        
    lir_polygon = Polygon(utils.vectorize_polygon_from_array(lir))
    
    inverted_lir_polygon = affinity.rotate(
        geom=lir_polygon, angle=-rotation_degree, origin=rotation_anchor
    )

    return inverted_lir_polygon

class LargestInscribedRectangle:
    def __init__(self, check_runtime: bool):
        self.solid = 1
        self.check_runtime = check_runtime

    def _get_lir_indices(self, binary_grid_shaped_lir: np.ndarray) -> np.ndarray:
        _, col_num = binary_grid_shaped_lir.shape
        height = np.zeros((col_num + 1,), dtype=int)
        max_area = 0
        top_left = ()
        bottom_right = ()

        for ri, row in enumerate(binary_grid_shaped_lir):
            height[:-1] = np.where(row == self.solid, height[:-1] + 1, 0)

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
    
    def _get_binary_grid_shaped_polygon(self, coordinates: np.ndarray, canvas_size: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            coordinates (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        
        binary_grid_shaped_polygon = np.zeros(canvas_size, np.uint8)
        cv2.fillPoly(binary_grid_shaped_polygon, [coordinates], 255)

        binary_grid_shaped_polygon = (binary_grid_shaped_polygon == 255).astype(np.uint8)

        return binary_grid_shaped_polygon
    
    @ray.remote
    def _get_each_lir(self, *lir_args):

        rotation_degree, rotation_anchor, binary_grid_shaped_polygon = lir_args

        top_left, bottom_right = self._get_lir_indices(binary_grid_shaped_polygon)
        
        top_left_row, top_left_col = top_left
        bottom_right_row, bottom_right_col = bottom_right
        
        lir = np.zeros_like(binary_grid_shaped_polygon)
        lir[top_left_row:bottom_right_row + 1, top_left_col:bottom_right_col + 1] = 1
            
        lir_polygon = Polygon(utils.vectorize_polygon_from_array(lir))
        
        inverted_lir_polygon = affinity.rotate(
            geom=lir_polygon, angle=-rotation_degree, origin=rotation_anchor
        )

        return inverted_lir_polygon
    
    @utils.runtime_calculator
    def _get_largest_inscribed_rectangle(
        self,
        random_coordinates: np.ndarray,
        canvas_size: np.ndarray,
    ) -> Polygon:
            
        lir_args = []
        
        random_coordinates_polygon = Polygon(random_coordinates)
        rotation_anchor = random_coordinates_polygon.centroid

        rotation_degree = 0
        while rotation_degree < 360:
            
            rotated_random_polygon = affinity.rotate(
                geom=random_coordinates_polygon,
                angle=rotation_degree,
                origin=rotation_anchor,
            ) 

            binary_grid_shaped_polygon = self._get_binary_grid_shaped_polygon(
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
            
            rotation_degree += self.lir_rotation_degree_interval
        
        lir_futures = [_get_each_lir.remote(*args) for args in lir_args]
        lirs = ray.get(lir_futures)

        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #     lirs = pool.starmap(self._get_each_lir, lir_args)

        return max(lirs, key=lambda p: p.area)