
import os
import torch
import numpy as np

from typing import List
from lirGAN.config import ModelConfig
from torch.utils.data import Dataset

class LirDataset(Dataset, ModelConfig):
    def __init__(self, data_path: str = None):
        
        self.lir_dataset: List[torch.tensor]
        self.lir_dataset = self._get_lir_dataset(self.DATA_PATH if data_path is None else data_path)
    
    def __len__(self) -> int:
        return len(self.lir_dataset)
    
    def __getitem__(self, index) -> torch.tensor:
        return self.lir_dataset[index].to(self.DEVICE)
        
    def _get_lir_dataset(self, data_path: str) -> List[torch.tensor]:
        """Load the lir dataset

        Args:
            data_path (str): path to load data

        Returns:
            List[torch.tensor]: Each tensor is (2, 500, 500)-shaped tensor. 
                                The first one is the input polygon and the second one is ground-truth rectangle
        """
        
        lir_dataset: List[torch.tensor]
        lir_dataset = []
        
        for file_name in os.listdir(data_path):
            lir_data_path = os.path.join(data_path, file_name)
            lir_dataset.append(
                torch.tensor(np.load(lir_data_path), dtype=torch.float32)
            )
        
        return lir_dataset


if __name__ == "__main__":
    lir_dataset = LirDataset()