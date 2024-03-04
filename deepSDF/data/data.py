import os
import torch
import numpy as np

from typing import List, Tuple
from torch.utils.data import Dataset
from deepSDF.config import Configuration


class SDFdataset(Dataset, Configuration):
    def __init__(self, data_path: str = None, slicer: int = np.inf):
        self.sdf_dataset: List[torch.Tensor]
        self.sdf_dataset = self._get_sdf_dataset(
            data_path=self.DATA_PATH if data_path is None else data_path, slicer=slicer
        )

    def __len__(self) -> int:
        return len(self.sdf_dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        xyz, sdf = self.sdf_dataset[index]

        return xyz.to(self.DEVICE), sdf.to(self.DEVICE)

    def _get_sdf_dataset(self, data_path: str, slicer: int) -> List[torch.Tensor]:
        """_summary_

        Args:
            data_path (str): _description_
            slicer (int): _description_

        Returns:
            List[torch.Tensor]: _description_
        """

        lir_dataset: List[torch.Tensor]
        lir_dataset = []

        for file_name in os.listdir(data_path):
            lir_data_path = os.path.join(data_path, file_name)
            lir_dataset.append(torch.tensor(np.load(lir_data_path), dtype=torch.float))

        if slicer < np.inf:
            return lir_dataset[:slicer]

        return lir_dataset
