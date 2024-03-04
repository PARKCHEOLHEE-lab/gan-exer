import os
import trimesh

from deepSDF.data import utils


class DataCreatorHelper:
    pass


class DataCreator:
    def __init__(self, n_points: int, raw_data_path: str) -> None:
        self.n_points = n_points
        self.raw_data_path = raw_data_path

    def _get_raw_meshes(self, raw_data_path: str) -> trimesh.Trimesh:
        """_summary_

        Args:
            raw_data_path (str): _description_

        Returns:
            trimesh.Trimesh: _description_
        """

        meshes = []
        for file in os.listdir(raw_data_path):
            meshes.append(utils.load_mesh(os.path.join(raw_data_path, file), normalize=True))

        return meshes

    def create(self) -> None:
        sdf = []  # noqa: F841
        xyz = []  # noqa: F841


if __name__ == "__main__":
    data_creator = DataCreator("")
    data_creator.create()
