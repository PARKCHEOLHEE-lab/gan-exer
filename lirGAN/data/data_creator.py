
from shapely.geometry import Polygon
from shapely import affinity

import cv2
import numpy as np
import largestinteriorrectangle as lir


class DataCreatorConfiguration:
    canvas_x = 400
    canvas_y = 700
    canvas_size = np.array([canvas_x, canvas_y])
    canvas_centroid = canvas_size / 2

class DataCreator(DataCreatorConfiguration):
    
    def __init__(self, creation_count: int) -> None:
        DataCreatorConfiguration.__init__(self)
        self.creation_count = creation_count
    
    

if __name__ == "__main__":
    
    data_creator = DataCreator(creation_count=1)
    
    polygon = np.array([[100, 100], [200, 200], [100, 200]])

    
    centralized_polygon = data_creator._get_centralized_coordinates(polygon)    
    normalized_polygon = data_creator._get_normalized_coordinates(centralized_polygon)    

    # res = normalize_and_centralize_polygon(polygon, data_creator.canvas_size)

    canvas = np.zeros((data_creator.canvas_size[0], data_creator.canvas_size[1], 3), dtype=np.uint8)
    

    # cv2.polylines(canvas, [res], 1, (0, 255, 0), 3)
    # cv2.imshow("Polygon Grid", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()