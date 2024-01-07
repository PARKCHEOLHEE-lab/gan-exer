
import cv2
import numpy as np

np.random.seed(0)


class LargestInscribedRectangle:
    def __init__(self, binary_grid_shaped_polygon: np.ndarray) -> None:
        self.binary_grid_shaped_polygon = binary_grid_shaped_polygon

class DataCreatorConfiguration:
    canvas_w_h = 700
    canvas_size = np.array([canvas_w_h, canvas_w_h])
    canvas_centroid = canvas_size / 2
    
    random_vertices_count_min = 5
    random_vertices_count_max = 25


class DataCreatorHelper(DataCreatorConfiguration):

    def __init__(self) -> None:
        DataCreatorConfiguration.__init__(self)
    
    def _get_random_coordinates(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
    
        vertices_count = np.random.randint(self.random_vertices_count_min, self.random_vertices_count_max)
        vertices = np.random.rand(vertices_count, 2)
        vertices_centroid = np.mean(vertices, axis=0)

        # To get a non-intersected polygon, sort vertices to CCW
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
    
    def _get_largest_inscribed_rectangle(self, binary_grid_shaped_polygon: np.ndarray):
        return
    
    def _get_vectorized_polygon_by_binary_grid(self, binary_grid_shaped_polygon: np.ndarray) -> np.ndarray:
        return


class DataCreator(DataCreatorHelper):
    
    def __init__(self, creation_count: int) -> None:
        DataCreatorHelper.__init__(self)

        self.creation_count = creation_count
    
    def create(self):
        
        for _ in range(self.creation_count):
            random_coordinates = self._get_random_coordinates()
            fitted_coordinates = self._get_fitted_coordinates(random_coordinates)
            binary_grid_shaped_polygon = self._get_binary_grid_shaped_polygon(fitted_coordinates)
            
            # lir = self._get_largest_inscribed_rectangle(random_coordinates)

        return

if __name__ == "__main__":

    from debugvisualizer.debugvisualizer import Plotter
    from shapely.geometry import Polygon, Point
    
    data_creator = DataCreator(creation_count=100)
    data_creator.create()
    
    # polygon = np.array([[100, 100], [200, 200], [100, 200]])

    # centralized_polygon = data_creator._get_centralized_coordinates(polygon)    
    # normalized_polygon = data_creator._get_normalized_coordinates(centralized_polygon)    

    # # res = normalize_and_centralize_polygon(polygon, data_creator.canvas_size)

    # canvas = np.zeros((data_creator.canvas_size[0], data_creator.canvas_size[1], 3), dtype=np.uint8)
    
    
    # # cv2.polylines(canvas, [res], 1, (0, 255, 0), 3)
    # # cv2.imshow("Polygon Grid", canvas)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
