import os
import ray
import multiprocessing
import numpy as np

from lirGAN.data.largest_inscribed_rectangle import LargestInscribedRectangle
from lirGAN.data import utils


class DataCreatorConfiguration:
    canvas_w_h = 256
    canvas_size = np.array([canvas_w_h, canvas_w_h])

    random_vertices_count_min = 3
    random_vertices_count_max = 25

    lir_rotation_degree_interval = 1.0


class DataCreatorHelper(DataCreatorConfiguration, LargestInscribedRectangle):
    def __init__(self) -> None:
        DataCreatorConfiguration.__init__(self)
        LargestInscribedRectangle.__init__(self)

    def _get_rotation_matrix(self, degree: float) -> np.ndarray:
        """Get matrix to rotate by a given degree

        Args:
            degree (float): degree to create a matrix

        Returns:
            np.ndarray: rotation matrix
        """

        cos_angle = np.cos(np.radians(degree))
        sin_angle = np.sin(np.radians(degree))

        return np.array(
            [
                [cos_angle, sin_angle],
                [-sin_angle, cos_angle],
            ]
        )

    def _get_random_coordinates(
        self, vertices_count_min: int, vertices_count_max: int, scale_factor: float = 1.0
    ) -> np.ndarray:
        """Generate non-intersected polygon randomly

        Args:
            vertices_count_min (int): random vertices count minimum value
            vertices_count_max (int): random vertices count maximum value
            scale_factor (float, optional): constant to scale. Defaults to 1.0.

        Returns:
            np.ndarray: random coordinates
        """

        vertices_count = np.random.randint(vertices_count_min, vertices_count_max)
        vertices = np.random.rand(vertices_count, 2)
        vertices_centroid = np.mean(vertices, axis=0)

        coordinates = sorted(vertices, key=lambda p, c=vertices_centroid: np.arctan2(p[1] - c[1], p[0] - c[0]))

        coordinates = np.array(coordinates)
        coordinates[:, 0] *= scale_factor
        coordinates[:, 1] *= scale_factor

        return coordinates

    def _get_fitted_coordinates(self, coordinates: np.ndarray, canvas_size: np.ndarray) -> np.ndarray:
        """Resize a given coordinates to the desired canvas size

        Args:
            coordinates (np.ndarray): polygon coordinates to resize

        Returns:
            np.ndarray: fitted polygon coordinates
        """

        min_x, min_y = np.min(coordinates, axis=0)
        max_x, max_y = np.max(coordinates, axis=0)

        scale_x = canvas_size[0] / (max_x - min_x)
        scale_y = canvas_size[1] / (max_y - min_y)
        scale = min(scale_x, scale_y)

        coordinates = (coordinates - np.array([min_x, min_y])) * scale

        new_min_x, new_min_y = np.min(coordinates, axis=0)
        new_max_x, new_max_y = np.max(coordinates, axis=0)

        offset_x = (canvas_size[0] - (new_max_x - new_min_x)) / 2 - new_min_x
        offset_y = (canvas_size[1] - (new_max_y - new_min_y)) / 2 - new_min_y

        coordinates = coordinates + np.array([offset_x, offset_y])

        return coordinates.astype(np.int32)


class DataCreator(DataCreatorHelper):
    def __init__(self, creation_count: int) -> None:
        DataCreatorHelper.__init__(self)

        self.creation_count = creation_count
        self.binpy_path = os.path.abspath(os.path.join(__file__, "..", "binpy"))

    def create(self) -> None:
        """Main function to create data for training"""

        ray.init(num_cpus=multiprocessing.cpu_count())

        if not os.path.isdir(self.binpy_path):
            os.mkdir(self.binpy_path)

        count = 0
        while count < self.creation_count:
            try:
                each_binpy_path = os.path.join(
                    self.binpy_path,
                    f"{count}.npy",
                )

                random_coordinates = self._get_fitted_coordinates(
                    self._get_random_coordinates(self.random_vertices_count_min, self.random_vertices_count_max),
                    self.canvas_size,
                )

                if os.path.isfile(each_binpy_path):
                    count += 1
                    continue

                lir = self._get_largest_inscribed_rectangle(
                    coordinates=random_coordinates,
                    canvas_size=self.canvas_size,
                )

                binary_grid_shaped_polygon = utils.get_binary_grid_shaped_polygon(
                    random_coordinates.astype(np.int32), self.canvas_size
                )

                binary_grid_shaped_lir = utils.get_binary_grid_shaped_polygon(
                    np.array(lir.exterior.coords, dtype=np.int32), self.canvas_size
                )

                np.save(each_binpy_path, np.array([binary_grid_shaped_polygon, binary_grid_shaped_lir]))

                count += 1

            except Exception as e:
                print(
                    f"""
                    creation failure:
                    {e}
                    """
                )

        print("creating done")


if __name__ == "__main__":
    from debugvisualizer.debugvisualizer import Plotter  # noqa: F401
    from shapely.geometry import Polygon, Point  # noqa: F401

    np.random.seed(0)

    data_creator = DataCreator(creation_count=4000)
    data_creator.create()
