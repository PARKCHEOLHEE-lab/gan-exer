import os
import torch
import torch.nn as nn
import torch.nn.init as init
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
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

        sdf_dataset: List[torch.Tensor] = []

        for file_name in os.listdir(data_path):
            if file_name.endswith(".npz"):
                sdf_data_path = os.path.join(data_path, file_name)
                data = np.load(sdf_data_path)
                xyz, sdf = data["xyz"], data["sdf"]
                sdf_dataset.append((torch.tensor(xyz, dtype=torch.float), torch.tensor(sdf, dtype=torch.float)))

        if slicer < np.inf:
            return sdf_dataset[:slicer]

        return sdf_dataset


class DeepSDF(nn.Module):
    def __init__(self, z_dim=256, data_shape=200):
        super().__init__()

        self.main_1 = nn.Sequential(
            nn.Linear(z_dim + 3, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 253),
        )

        self.main_2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Tanh(),
        )

        self.latent_vectors = nn.Parameter(torch.FloatTensor(data_shape, z_dim))

        # Initialize with a Gaussian distribution
        init.normal_(self.latent_vectors, mean=0, std=0.01)

    def forward(self, ind, xyz):
        latent_codes = self.latent_vectors[ind].repeat(xyz.shape[0], 1)

        cxyz_1 = torch.cat((latent_codes, xyz), dim=1)
        x1 = self.main_1(cxyz_1)

        # skip connecetion
        cxyz_2 = torch.cat((x1, cxyz_1), dim=1)
        x2 = self.main_2(cxyz_2)

        return x2

    @property
    def latent_vectors(self):
        return self.latent_vectors
