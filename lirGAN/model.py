
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

class LirGeometricLoss(nn.Module):
    def __init__(
        self, 
        bce_weight: float = 1.0, 
        diou_weight: float = 1.0, 
        area_weight: float = 1.0, 
        feasibility_weight: float = 1.0
    ):
        super().__init__()
        
        self.bce_weight = bce_weight 
        self.diou_weight = diou_weight 
        self.area_weight = area_weight 
        self.feasibility_weight = feasibility_weight
        
        self.bce_loss_function = nn.BCEWithLogitsLoss()

    @staticmethod
    def compute_diou_loss(generated_lir: torch.Tensor, target_lir: torch.Tensor, diou_weight: float) -> torch.Tensor:
        """compute the distance-IoU loss

        Args:
            generated_lir (torch.Tensor):  generated rectangle
            target_lir (torch.Tensor): target rectangle
            diou_weight (float): weight to multiply in the diou loss

        Returns:
            torch.Tensor: diou loss
        """
        
        intersection = torch.logical_and(generated_lir, target_lir).float()
        union = torch.logical_or(generated_lir, target_lir).float()
        iou_score = torch.sum(intersection) / torch.sum(union)
        
        generated_lir_centroid = torch.nonzero(generated_lir).float().mean(dim=0)
        target_lir_centroid = torch.nonzero(target_lir).float().mean(dim=0)

        distance = torch.norm(generated_lir_centroid - target_lir_centroid)

        diagonal = torch.norm(torch.tensor(generated_lir.shape).float())

        normalized_distance = (distance / diagonal) ** 2

        diou_loss = 1 - (iou_score - normalized_distance)

        return diou_loss * diou_weight
    
    @staticmethod
    def compute_area_loss(generated_lir: torch.Tensor, area_weight: float) -> torch.Tensor:
        """compute the area loss to maximize the size of a generated rectangle

        Args:
            generated_lir (torch.Tensor):  generated rectangle
            area_weight (float): weight to multiply in the area loss

        Returns:
            torch.Tensor: area loss
        """
        
        return -generated_lir.sum() * area_weight
    
    @staticmethod
    def compute_feasibility_loss(input_polygon: torch.Tensor, generated_lir: torch.Tensor, feasibility_weight: float) -> torch.Tensor:
        """compute the feasibility loss that checks the generated rectangle is within the input polygon

        Args:
            input_polygon (torch.Tensor): input polygon boundary to predict
            generated_lir (torch.Tensor): generated rectangle
            feasibility_weight (float): weight to multiply in the feasibility loss

        Returns:
            torch.Tensor: feasibility loss
        """

        infeasibility_mask = (generated_lir == 1) & (input_polygon != 1)
        feasibility_loss = infeasibility_mask.sum().float() / input_polygon.sum().float()
        
        return feasibility_loss * feasibility_weight
    
    def forward(self, input_polygons: torch.Tensor, generated_lirs: torch.Tensor, target_lirs: torch.Tensor) -> torch.Tensor:
        """compute total loss

        Args:
            input_polygons (torch.Tensor): input polygon boundaries to predict
            generated_lirs (torch.Tensor): generated rectangles
            target_lirs (torch.Tensor): target rectangles

        Returns:
            torch.Tensor: loss merged with bce, diou, area, feasibility
        """
        
        total_loss = 0
        for generated_lir, target_lir, input_polygon in zip(generated_lirs, target_lirs, input_polygons):
            bce_loss = self.bce_loss_function(generated_lir, target_lir)

            diou_loss = LirGeometricLoss.compute_diou_loss(generated_lir, target_lir, self.diou_weight)
            area_loss = LirGeometricLoss.compute_area_loss(generated_lir, self.area_weight)
            feasibility_loss = LirGeometricLoss.compute_feasibility_loss(input_polygon, generated_lir, self.feasibility_weight)
            
            loss = bce_loss + diou_loss + area_loss + feasibility_loss

            total_loss += loss

        return total_loss
    

if __name__ == "__main__":
    lir_dataset = LirDataset()
    
    lir_loss_function = LirGeometricLoss()
    
    input_polygons = torch.tensor(
        [
            [
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1],
            ],
        ]
    ).float()

    generated_lirs = torch.tensor(
        [
            [
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ],
    ).float()

    target_lirs = torch.tensor(
        [
            [
                [0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ],
    ).float()
    
    loss_function = LirGeometricLoss(
        bce_weight=1.0, 
        diou_weight=1.0, 
        area_weight=1.0, 
        feasibility_weight=10.0
    )

    loss_function(
        input_polygons=input_polygons, 
        generated_lirs=generated_lirs, 
        target_lirs=target_lirs
    )
    