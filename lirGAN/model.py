
import os
import torch
import torch.nn as nn
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

class LirGeometricalLoss(nn.Module):
    def __init__(self, bce_weight, diou_weight, area_weight, feasibility_weight):
        super().__init__()
        
        self.bce_weight = bce_weight 
        self.diou_weight = diou_weight 
        self.area_weight = area_weight 
        self.feasibility_weight = feasibility_weight
        
        self.bce_loss_function = nn.BCEWithLogitsLoss()

    @staticmethod
    def compute_diou_loss(generated_lir, target_lir, diou_weight):
        intersection = torch.logical_and(generated_lir, target_lir)
        union = torch.logical_or(generated_lir, target_lir)
        iou_score = torch.sum(intersection).float() / torch.sum(union).float()

        center_pred = torch.tensor([torch.mean(torch.nonzero(generated_lir, as_tuple=True)[i]) for i in range(2)])
        center_target = torch.tensor([torch.mean(torch.nonzero(target_lir, as_tuple=True)[i]) for i in range(2)])
        distance = torch.norm(center_pred - center_target, p=2)

        diagonal = torch.norm(torch.tensor(generated_lir.shape) - 1, p=2)
        normalized_distance = (distance / diagonal) ** 2

        diou = iou_score - normalized_distance

        return (1 - diou) * diou_weight
    
    @staticmethod
    def compute_area_loss(generated_lir, area_weight):
        return
    
    @staticmethod
    def _compute_feasibility_loss(generated_lir, input_polygon, feasibility_weight):
        return
    
    def forward(self):
        return

if __name__ == "__main__":
    lir_dataset = LirDataset()