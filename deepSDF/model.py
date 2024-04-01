import os
import time
import torch
import datetime
import numpy as np
import commonutils
import torch.nn as nn

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tqdm import tqdm
from typing import Tuple, Dict
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from deepSDF.config import Configuration
from torch.utils.data import Dataset, DataLoader, random_split
from deepSDF.reconstruct import Reconstructor


class SDFdataset(Dataset, Configuration):
    def __init__(self, data_path: str = Configuration.SAVE_DATA_PATH):
        self.sdf_dataset, self.cls_nums, self.cls_dict = self._get_sdf_dataset(data_path=data_path)

    def __len__(self) -> int:
        return len(self.sdf_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        xyz = self.sdf_dataset[index, :3]
        sdf = self.sdf_dataset[index, 3]
        cls = self.sdf_dataset[index, 4].long()

        return xyz.to(self.DEVICE), sdf.to(self.DEVICE), cls.to(self.DEVICE)

    def _get_sdf_dataset(self, data_path: str) -> Tuple[torch.Tensor, int, Dict[int, str]]:
        """Get dataset and pieces of information related to the dataset.

        Args:
            data_path (str): path to the dataset

        Returns:
            List[torch.Tensor]: dataset, number of classes, class dictionary
        """

        xyzs = torch.tensor([], dtype=torch.float)
        sdfs = torch.tensor([], dtype=torch.float)
        clss = torch.tensor([], dtype=torch.long)

        cls_dict = {}

        for file_name in os.listdir(data_path):
            if file_name.endswith(".npz"):
                npz_data_path = os.path.join(data_path, file_name)
                data = np.load(npz_data_path)

                xyz = data["xyz"]
                sdf = data["sdf"]
                cls = int(data["cls"])
                cls_name = str(data["cls_name"])

                if cls_name not in cls_dict:
                    cls_dict[cls] = cls_name

                if sum(xyzs.shape) == 0:
                    xyzs = torch.tensor(xyz, dtype=torch.float)
                    sdfs = torch.tensor(sdf, dtype=torch.float)
                    clss = torch.full((xyz.shape[0], 1), cls, dtype=torch.long)

                else:
                    xyzs = torch.vstack([xyzs, torch.tensor(xyz, dtype=torch.float)])
                    sdfs = torch.vstack([sdfs, torch.tensor(sdf, dtype=torch.float)])
                    clss = torch.vstack([clss, torch.full((xyz.shape[0], 1), cls, dtype=torch.long)])

        post_conditions = [
            xyzs.shape[0] == sdfs.shape[0] == clss.shape[0],
            xyzs.shape[1] == 3,
            sdfs.shape[1] == 1,
            clss.shape[1] == 1,
        ]

        assert all(post_conditions), "Data conditions are invalid."

        sdf_dataset = torch.hstack([xyzs, sdfs, clss])
        cls_nums = clss.max() + 1

        return sdf_dataset, cls_nums, cls_dict


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
        self.latent_codes.to(self.DEVICE)
        self.to(self.DEVICE)

    def forward(self, i, xyz, cxyz_1=None):
        if cxyz_1 is None:
            cxyz_1 = torch.cat((self.latent_codes[i], xyz), dim=1)

        x1 = self.main_1(cxyz_1)

        # skip connection
        cxyz_2 = torch.cat((x1, cxyz_1), dim=1)
        x2 = self.main_2(cxyz_2)

        return x2


class SDFdecoderTrainer(Reconstructor, Configuration):
    def __init__(
        self,
        sdf_dataset: Dataset,
        sdf_decoder: SDFdecoder,
        seed: int = Configuration.DEFAULT_SEED,
        pre_trained_path: str = "",
        is_debug_mode: bool = False,
        is_reconstruct_mode: bool = False,
    ):
        self.sdf_dataset = sdf_dataset
        self.sdf_decoder = sdf_decoder
        self.pre_trained_path = pre_trained_path
        self.is_debug_mode = is_debug_mode
        self.is_reconstruct_mode = is_reconstruct_mode

        self.is_valid_pre_trained_path = os.path.isdir(pre_trained_path) and len(os.listdir(pre_trained_path)) > 0

        if self.is_debug_mode:
            commonutils.add_debugvisualizer(globals())

        self.set_seed(seed)

        if not self.is_reconstruct_mode:
            self._set_summary_writer()
            self._set_dataloaders()
            self._set_loss_function()
            self._set_optimizers()
            self._set_weights()
            self._set_lr_schedulers()

    def _set_summary_writer(self) -> None:
        """Set all paths to log. If the pre-trained path exists, use it."""

        self.log_dir = os.path.join(self.LOG_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.is_valid_pre_trained_path:
            self.log_dir = self.pre_trained_path

        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        self.reconstruct_dir = os.path.join(self.log_dir, "reconstructed")
        if not os.path.exists(self.reconstruct_dir):
            os.mkdir(self.reconstruct_dir)

        self.state_dir = os.path.join(self.log_dir, "states")
        if not os.path.exists(self.state_dir):
            os.mkdir(self.state_dir)

        self.all_states_path = os.path.join(self.state_dir, "all_states.pth")
        if self.is_valid_pre_trained_path:
            self.all_states = torch.load(self.all_states_path)
            print(f"Set all pre-trained states from {self.all_states_path} \n")

        self.obj_path = os.path.join(self.reconstruct_dir, "reconstructed.obj")

    def _set_dataloaders(self) -> None:
        """Set dataloaders for training and validation by the ratios."""

        train_dataset, val_dataset = random_split(self.sdf_dataset, [self.TRAIN_DATASET_RATIO, self.VAL_DATASET_RATIO])

        self.sdf_train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        self.sdf_val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
        )

        print("Set all dataloaders")
        print(f"  TRAIN_DATASET_RATIO: {self.TRAIN_DATASET_RATIO}")
        print(f"  VAL_DATASET_RATIO: {self.VAL_DATASET_RATIO} \n")

    def _set_loss_function(self) -> None:
        """Set l1loss
        https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py#L378
        """

        self.loss_function = nn.L1Loss().to(self.DEVICE)

    def _set_optimizers(self) -> None:
        """Set all optimizers. If the pre-trained path exists, use them."""

        self.decoder_optimizer = torch.optim.Adam(self.sdf_decoder.parameters(), lr=self.LEARNING_RATE_MODEL)
        self.latent_optimizer = torch.optim.Adam([self.sdf_decoder.latent_codes], lr=self.LEARNING_RATE_LATENT)

        if self.is_valid_pre_trained_path:
            self.latent_optimizer.load_state_dict(self.all_states["optimizer_l"])
            self.decoder_optimizer.load_state_dict(self.all_states["optimizer_d"])
            print("Set all pre-trained optimizers \n")

    def _set_weights(self) -> None:
        """Set latent code weights. If the pre-trained path exists, use it."""

        nn.init.normal_(self.sdf_decoder.latent_codes, mean=0, std=0.01)

        if self.is_valid_pre_trained_path:
            self.sdf_decoder.load_state_dict(self.all_states["model_d"])
            self.sdf_decoder.latent_codes = nn.Parameter(self.all_states["latent_codes"])
            print("Set all pre-trained weights \n")

    def _set_lr_schedulers(self) -> None:
        """Set learning rate schedulers."""

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
        """Train each epoch.

        Args:
            sdf_decoder (SDFdecoder): model
            sdf_train_dataloader (DataLoader): dataloader for training
            decoder_optimizer (torch.optim.Optimizer): optimizer for decoder
            latent_optimizer (torch.optim.Optimizer): optimizer for latent codes
            loss_function (_Loss): loss function

        Returns:
            torch.Tensor: average loss for training
        """

        train_losses = []

        for xyz_batch, sdf_batch, cls_batch in tqdm(
            sdf_train_dataloader, desc=f"Training in `{epoch}th` epoch", leave=False
        ):
            sdf_batch = sdf_batch.unsqueeze(1)

            pred = sdf_decoder(cls_batch, xyz_batch)
            pred_clamped = torch.clamp(pred, -self.CLAMP_VALUE, self.CLAMP_VALUE)

            decoder_optimizer.zero_grad()
            latent_optimizer.zero_grad()

            l1_loss = loss_function(pred_clamped, sdf_batch)
            l1_loss.backward()

            decoder_optimizer.step()
            latent_optimizer.step()

            train_losses.append(l1_loss.item())

        return sum(train_losses) / len(train_losses)

    def _evaluate_each_epoch(
        self,
        sdf_decoder: SDFdecoder,
        sdf_val_dataloader: DataLoader,
        loss_function: _Loss,
        epoch: int,
    ) -> torch.Tensor:
        """Evaluate each epoch.

        Args:
            sdf_decoder (SDFdecoder): model
            sdf_val_dataloader (DataLoader): dataloader for validation
            loss_function (_Loss): loss function
            epoch (int): epoch

        Returns:
            torch.Tensor: average loss for validation
        """

        val_losses = []

        sdf_decoder.eval()
        with torch.no_grad():
            for xyz_batch, sdf_batch, cls_batch in tqdm(
                sdf_val_dataloader, desc=f"Evaluating in `{epoch}th` epoch", leave=False
            ):
                sdf_batch = sdf_batch.unsqueeze(1)

                pred = sdf_decoder(cls_batch, xyz_batch)
                pred_clamped = torch.clamp(pred, -self.CLAMP_VALUE, self.CLAMP_VALUE)

                l1_loss = loss_function(pred_clamped, sdf_batch)

                val_losses.append(l1_loss.item())

        sdf_decoder.train()

        return sum(val_losses) / len(val_losses)

    def train(self) -> None:
        """Train DeepSDF decoder."""

        best_loss = torch.inf
        start = 1

        if self.is_valid_pre_trained_path:
            start = self.all_states["epoch"] + 1
            best_loss = self.all_states["best_loss"]

        print(f"best_loss: {best_loss} \n")

        for epoch in range(start, self.EPOCHS + 1):
            start_time_per_epoch = time.time()

            avg_train_loss = self._train_each_epoch(
                sdf_decoder=self.sdf_decoder,
                sdf_train_dataloader=self.sdf_train_dataloader,
                decoder_optimizer=self.decoder_optimizer,
                latent_optimizer=self.latent_optimizer,
                loss_function=self.loss_function,
                epoch=epoch,
            )

            if epoch % self.LOG_INTERVAL == 0:
                avg_val_loss = self._evaluate_each_epoch(
                    sdf_decoder=self.sdf_decoder,
                    sdf_val_dataloader=self.sdf_val_dataloader,
                    loss_function=self.loss_function,
                    epoch=epoch,
                )

                self.reconstruct(
                    self.sdf_decoder, self.sdf_dataset.cls_dict, self.obj_path, epoch, normalize=False, map_z_to_y=True
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
                        "cls_dict": self.sdf_dataset.cls_dict,
                    }

                    torch.save(states, self.all_states_path)

                    print(f"Epoch: {epoch}th Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

                else:
                    states = torch.load(self.all_states_path)
                    states.update({"epoch": epoch})

                    torch.save(states, self.all_states_path)

                self.summary_writer.add_scalar("Loss/train", avg_train_loss, epoch)
                self.summary_writer.add_scalar("Loss/val", avg_val_loss, epoch)

            print(f"Epoch: {epoch}th epoch took {time.time() - start_time_per_epoch} seconds")
