
import cv2
import numpy as np


class DataCreatorConfiguration:
    canvas_w_h = 700
    canvas_size = np.array([canvas_w_h, canvas_w_h])
    canvas_centroid = canvas_size / 2


class DataCreatorHelper(DataCreatorConfiguration):

    def __init__(self) -> None:
        DataCreatorConfiguration.__init__(self)
    
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
    
    def _get_random_coordinates(self):
        
    
        vertices_count = np.random.randint(3, 50)

        points = np.random.rand(vertices_count, 2)

        return

class DataCreator(DataCreatorHelper):
    
    def __init__(self, creation_count: int) -> None:
        DataCreatorHelper.__init__(self)

        self.creation_count = creation_count
    
    def create(self):
        return

if __name__ == "__main__":
    
    data_creator = DataCreator(creation_count=1)
    
    # polygon = np.array([[100, 100], [200, 200], [100, 200]])

    # centralized_polygon = data_creator._get_centralized_coordinates(polygon)    
    # normalized_polygon = data_creator._get_normalized_coordinates(centralized_polygon)    

    # # res = normalize_and_centralize_polygon(polygon, data_creator.canvas_size)

    # canvas = np.zeros((data_creator.canvas_size[0], data_creator.canvas_size[1], 3), dtype=np.uint8)
    
    
    # # cv2.polylines(canvas, [res], 1, (0, 255, 0), 3)
    # # cv2.imshow("Polygon Grid", canvas)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()