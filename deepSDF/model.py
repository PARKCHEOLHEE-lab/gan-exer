import os
import torch
import trimesh
import datetime
import numpy as np
import commonutils
import torch.nn as nn


from tqdm import tqdm
from typing import List, Tuple
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from deepSDF.config import Configuration
from torch.utils.data import Dataset, DataLoader, random_split
from deepSDF import reconstruct


class SDFdataset(Dataset):
    def __init__(self, data_path: str = Configuration.SAVE_DATA_PATH):
        self.sdf_dataset, self.cls_nums = self._get_sdf_dataset(data_path=data_path)

    def __len__(self) -> int:
        return len(self.sdf_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        xyz = self.sdf_dataset[index, :3]
        sdf = self.sdf_dataset[index, 3]
        cls = self.sdf_dataset[index, 4].long()

        return xyz.to(Configuration.DEVICE), sdf.to(Configuration.DEVICE), cls.to(Configuration.DEVICE)

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

        return torch.hstack([xyzs, sdfs, clss]), len(os.listdir(data_path))


class SDFdecoder(nn.Module):
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
            nn.Linear(512, 512),
        )

        self.main_2 = nn.Sequential(
            nn.Linear(latent_size + 3 + 512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Tanh(),
        )

        self.latent_codes = nn.Parameter(torch.FloatTensor(cls_nums, latent_size))

        self.to(Configuration.DEVICE)
        self.latent_codes.to(Configuration.DEVICE)

    def forward(self, i, xyz):
        cxyz_1 = torch.cat((self.latent_codes[i], xyz), dim=1)
        x1 = self.main_1(cxyz_1)

        # skip connection
        cxyz_2 = torch.cat((x1, cxyz_1), dim=1)
        x2 = self.main_2(cxyz_2)

        return x2


class SDFdecoderTrainer(Configuration):
    def __init__(
        self,
        sdf_dataset: Dataset,
        sdf_decoder: SDFdecoder,
        seed: int = Configuration.DEFAULT_SEED,
        pre_trained_path: str = "",
        is_debug_mode: bool = False,
    ):
        self.sdf_dataset = sdf_dataset
        self.sdf_decoder = sdf_decoder
        self.pre_trained_path = pre_trained_path
        self.is_debug_mode = is_debug_mode

        self.has_pre_trained_path = os.path.isdir(pre_trained_path)

        if self.is_debug_mode:
            commonutils.add_debugvisualizer(globals())

        self.set_seed(seed)

        self._set_summary_writer()
        self._set_dataloaders()
        self._set_loss_function()
        self._set_optimizers()
        self._set_weights()
        self._set_lr_schedulers()

    def _set_summary_writer(self):
        """_summary_"""

        self.log_dir = os.path.join(Configuration.LOG_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.has_pre_trained_path:
            self.log_dir = self.pre_trained_path

        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        self.reconstruct_dir = Configuration.RECONSTRUCT_DIR
        if not os.path.exists(Configuration.RECONSTRUCT_DIR):
            os.mkdir(Configuration.RECONSTRUCT_DIR)

        self.state_dir = Configuration.STATE_DIR
        if not os.path.exists(Configuration.STATE_DIR):
            os.mkdir(Configuration.STATE_DIR)

        if self.has_pre_trained_path:
            self.all_states = torch.load(Configuration.ALL_STATES_PATH)

        self.obj_path = Configuration.RECONSTRUCT_PATH

    def _set_dataloaders(self):
        train_dataset, val_dataset = random_split(
            self.sdf_dataset, [Configuration.TRAIN_DATASET_RATIO, Configuration.VAL_DATASET_RATIO]
        )

        self.sdf_train_dataloader = DataLoader(
            train_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        self.sdf_val_dataloader = DataLoader(
            val_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
        )

    def _set_loss_function(self):
        """_summary_"""

        self.loss_function = nn.L1Loss().to(Configuration.DEVICE)

    def _set_optimizers(self):
        """_summary_"""

        self.decoder_optimizer = torch.optim.Adam(self.sdf_decoder.parameters(), lr=Configuration.LEARNING_RATE_MODEL)
        self.latent_optimizer = torch.optim.Adam([self.sdf_decoder.latent_codes], lr=Configuration.LEARNING_RATE_LATENT)

        if self.has_pre_trained_path:
            self.latent_optimizer.load_state_dict(self.all_states["optimizer_l"])
            self.decoder_optimizer.load_state_dict(self.all_states["optimizer_d"])

    def _set_weights(self):
        nn.init.normal_(self.sdf_decoder.latent_codes, mean=0, std=0.01)

        if self.has_pre_trained_path:
            self.sdf_decoder.load_state_dict(self.all_states["model_d"])
            self.sdf_decoder.latent_codes = nn.Parameter(self.all_states["latent_codes"])

    def _set_lr_schedulers(self) -> None:
        self.scheduler_d = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, patience=5, verbose=True, factor=0.1)
        self.scheduler_l = lr_scheduler.ReduceLROnPlateau(self.latent_optimizer, patience=5, verbose=True, factor=0.1)

    def _train_each_epoch(
        self,
        sdf_decoder: SDFdecoder,
        sdf_train_dataloader: DataLoader,
        decoder_optimizer: torch.optim.Optimizer,
        latent_optimizer: torch.optim.Optimizer,
        loss_function: _Loss,
        epoch: int,
    ) -> torch.Tensor:
        """_summary_

        Args:
            sdf_decoder (SDFdecoder): _description_
            sdf_train_dataloader (DataLoader): _description_
            decoder_optimizer (torch.optim.Optimizer): _description_
            latent_optimizer (torch.optim.Optimizer): _description_
            loss_function (_Loss): _description_

        Returns:
            torch.Tensor: _description_
        """

        losses = []
        for xyz_batch, sdf_batch, cls_batch in tqdm(sdf_train_dataloader, desc=f"Training in `{epoch}th` epoch"):
            sdf_batch = sdf_batch.unsqueeze(1)

            pred = sdf_decoder(cls_batch, xyz_batch)
            pred_clamped = torch.clamp(pred, -Configuration.CLAMP_VALUE, Configuration.CLAMP_VALUE)

            decoder_optimizer.zero_grad()
            latent_optimizer.zero_grad()

            l1_loss = loss_function(pred_clamped, sdf_batch)
            l1_loss.backward()

            decoder_optimizer.step()
            latent_optimizer.step()

            losses.append(l1_loss.item())

        return sum(losses) / len(losses)

    def _evaluate_each_epoch(
        self, sdf_decoder: SDFdecoder, sdf_val_dataloader: DataLoader, loss_function: _Loss, obj_path: str, epoch: int
    ) -> torch.Tensor:
        """_summary_

        Args:
            sdf_decoder (SDFdecoder): _description_
            sdf_val_dataloader (DataLoader): _description_
            epoch (int): _description_

        Returns:
            torch.Tensor: _description_
        """

        losses = []

        sdf_decoder.eval()
        with torch.no_grad():
            for xyz_batch, sdf_batch, cls_batch in tqdm(sdf_val_dataloader, desc=f"Evaluating in `{epoch}th` epoch"):
                sdf_batch = sdf_batch.unsqueeze(1)

                pred = sdf_decoder(cls_batch, xyz_batch)
                pred_clamped = torch.clamp(pred, -Configuration.CLAMP_VALUE, Configuration.CLAMP_VALUE)

                l1_loss = loss_function(pred_clamped, sdf_batch)

                losses.append(l1_loss.item())

            coords, grid_size_axis = reconstruct.ReconstructorHelper.get_volume_coords(
                resolution=int(Configuration.RECONSTRUCT_RESOLUTION)
            )
            coords.to(Configuration.DEVICE)
            coords_batches = torch.split(coords, coords.shape[0] // 1000)

            sdf = torch.tensor([]).to(Configuration.DEVICE)

            for coords_batch in tqdm(coords_batches, desc=f"Reconstructing in `{epoch}th` epoch"):
                cls = torch.tensor([0] * coords_batch.shape[0], dtype=torch.long).to(Configuration.DEVICE)
                pred = sdf_decoder(cls, coords_batch)
                if sum(sdf.shape) == 0:
                    sdf = pred
                else:
                    sdf = torch.vstack([sdf, pred])

            vertices, faces = reconstruct.ReconstructorHelper.extract_mesh(grid_size_axis=grid_size_axis, sdf=sdf)

            if vertices is not None and faces is not None:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
                mesh.export(obj_path.replace(".obj", f"_{epoch}.obj"))

        sdf_decoder.train()

        return sum(losses) / len(losses)

    def train(self) -> None:
        """_summary_"""

        best_loss = torch.inf

        start = 1
        if self.has_pre_trained_path:
            start = self.all_states["epoch"] + 1

        for epoch in range(start, Configuration.EPOCHS + 1):
            avg_train_loss = self._train_each_epoch(
                sdf_decoder=self.sdf_decoder,
                sdf_train_dataloader=self.sdf_train_dataloader,
                decoder_optimizer=self.decoder_optimizer,
                latent_optimizer=self.latent_optimizer,
                loss_function=self.loss_function,
                epoch=epoch,
            )

            if epoch % Configuration.LOG_INTERVAL == 0:
                avg_val_loss = self._evaluate_each_epoch(
                    sdf_decoder=self.sdf_decoder,
                    sdf_val_dataloader=self.sdf_val_dataloader,
                    loss_function=self.loss_function,
                    obj_path=self.obj_path,
                    epoch=epoch,
                )

                self.scheduler_d.step(avg_val_loss)
                self.scheduler_l.step(avg_val_loss)

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss

                    states = {
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "model_d": self.sdf_decoder.state_dict(),
                        "latent_codes": self.sdf_decoder.latent_codes,
                        "optimizer_l": self.latent_optimizer.state_dict(),
                        "optimizer_d": self.decoder_optimizer.state_dict(),
                        "configuration": {k: v for k, v in Configuration()},
                    }

                    torch.save(states, Configuration.ALL_STATES_PATH)

                    print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

                else:
                    states = torch.load(Configuration.ALL_STATES_PATH)
                    states.update({"epoch": epoch})

                self.summary_writer.add_scalar("Loss/train", avg_train_loss, epoch)
                self.summary_writer.add_scalar("Loss/val", avg_val_loss, epoch)


if __name__ == "__main__":
    sdf_dataset = SDFdataset()
    sdf_decoder = SDFdecoder(cls_nums=sdf_dataset.cls_nums, latent_size=Configuration.LATENT_SIZE)
    sdf_trainer = SDFdecoderTrainer(
        sdf_dataset=sdf_dataset,
        sdf_decoder=sdf_decoder,
        is_debug_mode=False,
        seed=77777,
        pre_trained_path=r"deepSDF\runs\2024-03-21_20-38-40",
    )

    sdf_trainer.train()
