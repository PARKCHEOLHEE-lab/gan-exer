import os
import torch
import torch.nn as nn
import numpy as np
import commonutils

from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from deepSDF.config import Configuration


class SDFdataset(Dataset, Configuration):
    def __init__(self, data_path: str = Configuration.SAVE_DATA_PATH):
        self.sdf_dataset: List[torch.Tensor]
        self.sdf_dataset = self._get_sdf_dataset(data_path=data_path)

    def __len__(self) -> int:
        return len(self.sdf_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        xyz = self.sdf_dataset[index, :3]
        sdf = self.sdf_dataset[index, 3]
        cls = self.sdf_dataset[index, 4]

        return xyz.to(self.DEVICE), sdf.to(self.DEVICE), cls.to(self.DEVICE)

    def _get_sdf_dataset(self, data_path: str) -> List[torch.Tensor]:
        """_summary_

        Args:
            data_path (str): _description_
            slicer (int): _description_

        Returns:
            List[torch.Tensor]: _description_
        """

        xyzs = torch.tensor([], dtype=torch.float)
        sdfs = torch.tensor([], dtype=torch.float)
        clss = torch.tensor([], dtype=torch.long)

        for file_name in os.listdir(data_path):
            if file_name.endswith(".npz"):
                sdf_data_path = os.path.join(data_path, file_name)
                data = np.load(sdf_data_path)

                xyz = data["xyz"]
                sdf = data["sdf"]
                cls = data["cls"]

                if sum(xyzs.shape) == 0:
                    xyzs = torch.tensor(xyz, dtype=torch.float)
                    sdfs = torch.tensor(sdf, dtype=torch.float)
                    clss = torch.full((xyz.shape[0], 1), int(cls), dtype=torch.long)

                else:
                    xyzs = torch.vstack([xyzs, torch.tensor(xyz, dtype=torch.float)])
                    sdfs = torch.vstack([sdfs, torch.tensor(sdf, dtype=torch.float)])
                    clss = torch.vstack([clss, torch.full((xyz.shape[0], 1), int(cls), dtype=torch.long)])

        assert xyzs.shape[0] == sdfs.shape[0] == clss.shape[0], "`xyzs`, `sdfs`, `clss` shape must be the same"
        assert xyzs.shape[1] == 3, "The shape of `xyzs` must be (n, 3)"
        assert sdfs.shape[1] == 1, "The shape of `sdfs` must be (n, 1)"
        assert clss.shape[1] == 1, "The shape of `clss` must be (n, 1)"

        return torch.hstack([xyzs, sdfs, clss])


class SDFdecoder(nn.Module, Configuration):
    def __init__(self, cls_nums: int, latent_size: int = Configuration.LATENT_SIZE):
        super().__init__()

        self.main_1 = nn.Sequential(
            nn.Linear(latent_size + 3, 512),
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

        self.latent_codes = nn.Parameter(torch.FloatTensor(cls_nums, latent_size))
        nn.init.normal_(self.latent_codes, mean=0, std=0.01)

        self.to(self.DEVICE)

    def forward(self, ind, xyz):
        latent_codes = self.latent_codes[ind].repeat(xyz.shape[0], 1)

        cxyz_1 = torch.cat((latent_codes, xyz), dim=1)
        x1 = self.main_1(cxyz_1)

        # skip connection
        cxyz_2 = torch.cat((x1, cxyz_1), dim=1)
        x2 = self.main_2(cxyz_2)

        return x2

    @property
    def latent_codes(self):
        return self.latent_codes


class SDFdecoderTrainer:
    def __init__(self, sdf_dataloader: DataLoader, is_debug_mode: bool = False):
        self.sdf_dataloader = sdf_dataloader
        self.is_debug_mode = is_debug_mode

        if self.is_debug_mode:
            commonutils.add_debugvisualizer(globals())

    def train(self):
        # for xyz, sdf, cls in self.sdf_dataloader:
        #     latent_codes = self.decoder.latent_codes[cls]
        # pass
        return

    def evaluate(self):
        return


if __name__ == "__main__":
    sdf_dataset = SDFdataset()
    sdf_dataloader = DataLoader(
        sdf_dataset,
        batch_size=Configuration.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    trainer = SDFdecoderTrainer(
        sdf_dataloader=sdf_dataloader,
        is_debug_mode=True,
    )

    print()

    # batch_size = 1
    # latent_size = 128

    # latent_codes = nn.Parameter(torch.FloatTensor(batch_size, latent_size)).to("cuda")
    # nn.init.normal_(latent_codes, mean=0, std=0.01)

    # xyz, sdf = dataset[0]

    # xyz = xyz.unsqueeze(0)

    # # latent_codes_batch = self.latent_codes[latent_classes_batch.view(-1)]    # shape (batch_size, 128)

    # # x = torch.hstack((latent_codes_batch, coords))                  # shape (batch_size, 131)

    # a=1

    # sdf_decder = SDFdecoder(3)
    # sdf_decder(0, xyz)
