import os
import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss

import numpy as np

from utils import Utils
from config import Config


class BinvoxDataset(Dataset, Config):
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.binvox_data_list: List[torch.tensor]
        self.binvox_data_list = self._get_binvox_data_list()
    
    def __len__(self) -> int:
        return len(self.binvox_data_list)
    
    def __getitem__(self, index) -> torch.tensor:
        return self.binvox_data_list[index].to(self.DEVICE)
        
    def _get_binvox_data_list(self) -> List[torch.tensor]:
        binvox_data_list = []
        
        for file_name in os.listdir(self.data_path):
            each_file_path = os.path.join(self.data_path, file_name)
            
            for data_name in os.listdir(each_file_path):
                each_data_path = os.path.join(each_file_path, data_name)
            
                model = Utils.get_binvox_model(each_data_path)
                binvox_data_list.append(torch.tensor(model.data, dtype=torch.float32))
        
        return binvox_data_list


class WeightsInitializer:
    @staticmethod
    def initialize_weights_xavier(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    @staticmethod
    def initialize_weights_he(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Generator(nn.Module, Config):
    def __init__(self, z_dim, init_out_channels: int = None):
        super().__init__()
        
        out_channels_0 = self.GENERATOR_INIT_OUT_CHANNELS if init_out_channels is None else init_out_channels
        out_channels_1 = int(out_channels_0 / 2)
        out_channels_2 = int(out_channels_1 / 2)

        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            
            nn.ConvTranspose3d(z_dim, out_channels_0, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels_0),
            nn.ReLU(True),
            # state size. (256) x 4 x 4 x 4
            
            nn.ConvTranspose3d(out_channels_0, out_channels_1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels_1),
            nn.ReLU(True),
            # state size. (128) x 8 x 8 x 8
            
            nn.ConvTranspose3d(out_channels_1, out_channels_2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels_2),
            nn.ReLU(True),
            # state size. (64) x 16 x 16 x 16
            
            nn.ConvTranspose3d(out_channels_2, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
            # state size. (1) x 32 x 32 x 32
        )
        
        self.to(self.DEVICE)
        
    def forward(self, x):
        return self.main(x)
    
    
class Discriminator(nn.Module, Config):
    def __init__(self, init_out_channels: int = None):
        super().__init__()
        
        out_channels_0 = self.DISCRIMINATOR_INIT_OUT_CHANNELS if init_out_channels is None else init_out_channels
        out_channels_1 = out_channels_0 * 2
        out_channels_2 = out_channels_1 * 2

        self.main = nn.Sequential(
            # input is (1) x 32 x 32 x 32
            
            nn.Conv3d(1, out_channels_0, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 16 x 16 x 16

            nn.Conv3d(out_channels_0, out_channels_1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels_1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128) x 8 x 8 x 8

            nn.Conv3d(out_channels_1, out_channels_2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels_2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 4 x 4 x 4
            
            nn.Conv3d(out_channels_2, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. (1) x 1 x 1 x 1
        )
        
        self.to(self.DEVICE)
        
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
    

class MassGANTrainer(Config, WeightsInitializer):
    def __init__(
        self, 
        generator: Generator,
        discriminator: Discriminator,
        dataloader: DataLoader,
        epochs: int, 
        loss_function: _Loss, 
        learning_rate: float,
        seed: int,
        apply_weights: str,
    ):  
        self.set_seed(seed)
        

        generated_dir = self.DATA_GENERATED_DIR_MERGED
        if not os.path.isdir(generated_dir):
            os.mkdir(generated_dir)

        self.generated_datetiime_dir = os.path.join(
            generated_dir, str(datetime.datetime.now()).replace(":", "-")
        )
        
        if not os.path.isdir(self.generated_datetiime_dir):
            os.mkdir(self.generated_datetiime_dir)
        
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.epochs = epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate

        if apply_weights == self.XAVIER:
            self.generator.apply(self.initialize_weights_xavier)
            self.discriminator.apply(self.initialize_weights_xavier)
        elif apply_weights == self.HE:
            self.generator.apply(self.initialize_weights_he)
            self.discriminator.apply(self.initialize_weights_he)
        
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate
        )
        
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )
        
    def _get_noise(self, size: Tuple[int] = (Config.BATCH_SIZE, Config.Z_DIM, 1, 1, 1)):
        return torch.randn(size).to(self.DEVICE)
        
    def _train_discriminator(self, real_data: torch.tensor) -> torch.tensor:
        noise = self._get_noise()
        fake_mass = self.generator(noise)
        
        real_d = self.discriminator(real_data.unsqueeze(1))
        fake_d = self.discriminator(fake_mass)

        loss_real_d = self.loss_function(real_d, torch.ones_like(real_d))
        loss_fake_d = self.loss_function(fake_d, torch.zeros_like(fake_d))
        loss_d = loss_real_d + loss_fake_d
        
        self.discriminator.zero_grad()
        loss_d.backward()
        self.discriminator_optimizer.step()
        
        return loss_d
        
    def _train_generator(self) -> torch.tensor:
        noise = self._get_noise()
        fake_mass = self.generator(noise)
        
        fake_d = self.discriminator(fake_mass)
        
        loss_g = self.loss_function(fake_d, torch.ones_like(fake_d))
        
        self.generator.zero_grad()
        loss_g.backward()
        self.generator_optimizer.step()
        
        return loss_g
    
    def _evaluate(self, evaluate_batch_size: int, epoch : int, save_npy: bool = False, plot_voxels=False) -> None:
        self.generator.eval()
        self.discriminator.eval()
                
        with torch.no_grad():
            noise = self._get_noise(size=(evaluate_batch_size, self.Z_DIM, 1, 1, 1))
            generated_binvox_mass = self.generator(noise).squeeze(1).cpu().numpy() > 0.5
            
            if save_npy:
                npy_save_name = os.path.join(self.generated_datetiime_dir, f"generated_samples_epoch_{epoch}.npy")
                np.save(npy_save_name, generated_binvox_mass)

            mass_data_list = [data.squeeze() for data in generated_binvox_mass]
            Utils.plot_binvox(data_list=mass_data_list, plot_voxels=plot_voxels)

        self.generator.train()
        self.discriminator.train()
        
    
    def train(self) -> None:
        losses_g = []
        losses_d = []

        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(self.generator_optimizer, patience=5, factor=0.1)
        scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(self.discriminator_optimizer, patience=5, factor=0.1)
        
        for epoch in range(1, self.epochs + 1):
            
            print(f"-------------------- epoch: {epoch}/{self.epochs} running")
            
            losses_g_per_epoch = []
            losses_d_per_epoch = []
            
            for real_data in self.dataloader:
                loss_d = self._train_discriminator(real_data=real_data)
                loss_g = self._train_generator()
                losses_g_per_epoch.append(loss_g.item())
                losses_d_per_epoch.append(loss_d.item())
            
            avg_loss_g = np.mean(losses_g_per_epoch)
            avg_loss_d = np.mean(losses_d_per_epoch)
            losses_g.append(avg_loss_g)
            losses_d.append(avg_loss_d)
            
            scheduler_g.step(avg_loss_g)
            scheduler_d.step(avg_loss_d)
            
            print(
                f""""--------------------{epoch}/{self.epochs} Loss status:
                loss g: {avg_loss_g.item()}
                loss d: {avg_loss_d.item()}
                """
                )
            
            self._evaluate(evaluate_batch_size=self.BATCH_SIZE_TO_EVALUATE, epoch=epoch, save_npy=True, plot_voxels=True)
                
            if epoch % self.LOG_INTERVAL == 0:
                print(f"pth saving at {epoch}/{self.epochs}")
                torch.save(self.generator.state_dict(), os.path.join(self.PTHS_DIR, f"generator_epoch_{epoch}.pth"))
                torch.save(self.discriminator.state_dict(), os.path.join(self.PTHS_DIR, f"discriminator_epoch_{epoch}.pth"))

            print(f"epoch: {epoch}/{self.epochs} terminating --------------------")
            print()
            
                
