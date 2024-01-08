
import cv2
import numpy as np
from lirGAN.data.largest_inscribed_rectangle import LargestInscribedRectangle

np.random.seed(0)

class DataCreatorConfiguration:
    canvas_w_h = 700
    canvas_size = np.array([canvas_w_h, canvas_w_h])
    canvas_centroid = canvas_size / 2
    
    random_vertices_count_min = 5
    random_vertices_count_max = 25
    
    lir_rotation_degree_interval = 1.0


class DataCreatorHelper(DataCreatorConfiguration, LargestInscribedRectangle):

    def __init__(self) -> None:
        DataCreatorConfiguration.__init__(self)
        LargestInscribedRectangle.__init__(self)
        
    def _get_rotation_matrix(self, degree: float) -> np.ndarray:
        """_summary_

        Args:
            degree (float): _description_

        Returns:
            np.ndarray: _description_
        """
        
        cos_angle = np.cos(np.radians(degree))
        sin_angle = np.sin(np.radians(degree))
        
        return np.array(
            [
                [cos_angle, sin_angle],
                [-sin_angle, cos_angle],
            ]
        )
    
    def _get_random_coordinates(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
    
        vertices_count = np.random.randint(self.random_vertices_count_min, self.random_vertices_count_max)
        vertices = np.random.rand(vertices_count, 2)
        vertices_centroid = np.mean(vertices, axis=0)

        coordinates = sorted(
            vertices, key=lambda p, c=vertices_centroid: np.arctan2(p[1] - c[1], p[0] - c[0])
        )

        return np.array(coordinates)
    
    def _get_fitted_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            coordinates (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        
        min_x, min_y = np.min(coordinates, axis=0)
        max_x, max_y = np.max(coordinates, axis=0)

        scale_x = self.canvas_size[0] / (max_x - min_x)
        scale_y = self.canvas_size[1] / (max_y - min_y)
        scale = min(scale_x, scale_y)

        coordinates = (coordinates - np.array([min_x, min_y])) * scale

        new_min_x, new_min_y = np.min(coordinates, axis=0)
        new_max_x, new_max_y = np.max(coordinates, axis=0)

        offset_x = (self.canvas_size[0] - (new_max_x - new_min_x)) / 2 - new_min_x
        offset_y = (self.canvas_size[1] - (new_max_y - new_min_y)) / 2 - new_min_y

        coordinates = coordinates + np.array([offset_x, offset_y])

        return coordinates.astype(np.int32)
    
    def _get_binary_grid_shaped_polygon(self, coordinates: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            coordinates (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        
        binary_grid_shaped_polygon = np.zeros(self.canvas_size, np.uint8)
        cv2.fillPoly(binary_grid_shaped_polygon, [coordinates], 255)

        binary_grid_shaped_polygon = (binary_grid_shaped_polygon == 255).astype(np.uint8)

        return binary_grid_shaped_polygon
    
    def _get_vectorized_polygon_by_binary_grid(self, binary_grid_shaped_polygon: np.ndarray) -> np.ndarray:
        return


class DataCreator(DataCreatorHelper):
    
    def __init__(self, creation_count: int) -> None:
        DataCreatorHelper.__init__(self)

        self.creation_count = creation_count
    
    def create(self):
        
        for _ in range(self.creation_count):
            random_coordinates = self._get_random_coordinates()
            
            lir_args_list = []
            rotation_degree = 0

            while rotation_degree < 360:
                
                rotated_coordinates = random_coordinates @ self._get_rotation_matrix(rotation_degree)
                fitted_coordinates = self._get_fitted_coordinates(rotated_coordinates)

                binary_grid_shaped_polygon = self._get_binary_grid_shaped_polygon(fitted_coordinates)
                
                lir_args = [binary_grid_shaped_polygon, rotation_degree]
                lir = self._get_each_lir(lir_args)
                
                # color_grid = cv2.cvtColor(lir.astype(np.float32), cv2.COLOR_GRAY2BGR)

                # # Assign colors (here, white for 1, black for 0)
                # color_grid[lir == 1] = [255, 255, 255]  # White
                # color_grid[lir == 0] = [0, 0, 0]        # Black

                # # Display the image
                # cv2.imshow('Binary Grid', color_grid)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # lir_args = (binary_grid_shaped_polygon, )

                # top_left, bottom_right = self._get_lir_indices(binary_grid_shaped_polygon)

                # canvas = np.zeros((*binary_grid_shaped_polygon.shape, 3), dtype=np.uint8)
                # cv2.polylines(canvas, [fitted_coordinates], 1, (0, 255, 0), 1)
                # cv2.imshow("Polygon Grid", canvas)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                rotation_degree += self.lir_rotation_degree_interval
    
    
            

            # largest_incribed_rectangle = self._get_largest_inscribed_rectangle(binary_grid_shaped_polygon, self.lir_rotation_interval)

        return


if __name__ == "__main__":

    from debugvisualizer.debugvisualizer import Plotter
    from shapely.geometry import Polygon, Point
    
    data_creator = DataCreator(creation_count=100)
    data_creator.create()