import numpy as np
from multiprocessing import Pool, cpu_count


class LargestInscribedRectangle:
    def __init__(self):
        self.solid = 1
    
    def _get_lir_indices(self, binary_grid_shaped_lir: np.ndarray) -> np.ndarray:

        _, col_num = len(binary_grid_shaped_lir), len(binary_grid_shaped_lir[0])
        height = [0] * (col_num + 1)
        max_area = 0
        top_left = ()
        bottom_right = ()

        for ri, row in enumerate(binary_grid_shaped_lir):
            for hi in range(col_num):
                height[hi] = (
                    height[hi] + 1
                    if row[hi] == self.solid
                    else 0
                )

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
        
    def _get_each_lir(self, lir_args):

        binary_grid_shaped_polygon, rotation_degree = lir_args

        top_left, bottom_right = self._get_lir_indices(binary_grid_shaped_polygon)
        
        top_left_row, top_left_col = top_left
        bottom_right_row, bottom_right_col = bottom_right
        
        lir = np.zeros_like(binary_grid_shaped_polygon)
        lir[top_left_row:bottom_right_row + 1, top_left_col:bottom_right_col + 1] = 1

        return lir
        
    def _get_largest_inscribed_rectangle(self, binary_grid_shaped_polygon: np.ndarray, lir_rotation_interval: float) -> np.ndarray:
        
        # with Pool(processes=cpu_count()) as pool:
        #     x_squared = pool.starmap(self._get_each_lir, lir_args)
            
        #     self._get_each_lir()
        
        return