import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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


class Generator(nn.Module, WeightsInitializer, Config):
    def __init__(self, z_dim, init_out_channels: int = None):
        super().__init__()
        
        out_channels_0 = self.GENERATOR_INIT_OUT_CHANNELS if init_out_channels is None else init_out_channels
        out_channels_1 = int(out_channels_0 / 2)
        out_channels_2 = int(out_channels_1 / 2)

        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            
            nn.ConvTranspose3d(z_dim, out_channels_0, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            # state size. (256) x 4 x 4 x 4
            
            nn.ConvTranspose3d(out_channels_0, out_channels_1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            # state size. (128) x 8 x 8 x 8
            
            nn.ConvTranspose3d(out_channels_1, out_channels_2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            # state size. (64) x 16 x 16 x 16
            
            nn.ConvTranspose3d(out_channels_2, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
            # state size. (1) x 32 x 32 x 32
        )
        
        self.apply(self.initialize_weights_xavier)
        self.to(self.DEVICE)
        
    def forward(self, x):
        return self.main(x)
    
    
class Discriminator(nn.Module, WeightsInitializer, Config):
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
        
        self.apply(self.initialize_weights_xavier)
        self.to(self.DEVICE)

        
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)