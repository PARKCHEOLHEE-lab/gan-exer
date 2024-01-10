
import cv2
import numpy as np
import multiprocessing

from lirGAN.data.largest_inscribed_rectangle import LargestInscribedRectangle
from lirGAN.data import utils

from shapely.geometry import Polygon
from shapely import affinity

np.random.seed(0)

class DataCreatorConfiguration:
    canvas_w_h = 700
    canvas_size = np.array([canvas_w_h, canvas_w_h])
    canvas_centroid = canvas_size / 2
    
    random_vertices_count_min = 5
    random_vertices_count_max = 25
    
    lir_rotation_degree_interval = 1.0


class DataCreatorHelper(DataCreatorConfiguration, LargestInscribedRectangle):

    def __init__(self, check_runtime: bool) -> None:
        DataCreatorConfiguration.__init__(self)
        LargestInscribedRectangle.__init__(self, check_runtime)
        
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
    
    def _get_random_coordinates(self, scale_factor: float = 1.0) -> np.ndarray:
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
        
        coordinates = np.array(coordinates)
        coordinates[:, 0] *= scale_factor
        coordinates[:, 1] *= scale_factor

        return coordinates
    
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
    

class DataCreator(DataCreatorHelper):
    
    def __init__(self, creation_count: int, check_runtime: bool = False) -> None:
        DataCreatorHelper.__init__(self, check_runtime)

        self.creation_count = creation_count
    
    def create(self):
        import time
        
        for _ in range(self.creation_count):
            
            random_coordinates = self._get_random_coordinates(scale_factor=self.canvas_w_h)
            
            lir = self._get_largest_inscribed_rectangle(
                random_coordinates=random_coordinates,
                canvas_size=self.canvas_size,
            )
            
            a=1
            
        return


if __name__ == "__main__":

    from debugvisualizer.debugvisualizer import Plotter
    from shapely.geometry import Polygon, Point
    
    data_creator = DataCreator(creation_count=1, check_runtime=True)
    data_creator.create()