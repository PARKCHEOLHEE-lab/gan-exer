import os
import torch
import random
import numpy as np


class Configuration:
    """Configuration related to the DeepSDF model"""

    DATA_PATH = os.path.join("")

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    DEFAULT_SEED = 777

    @staticmethod
    def set_seed(seed: int = DEFAULT_SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        print("Seeds status:")
        print(f"  Seeds set for torch        : {torch.initial_seed()}")
        print(f"  Seeds set for torch on GPU : {torch.cuda.initial_seed()}")
        print(f"  Seeds set for numpy        : {seed}")
        print(f"  Seeds set for random       : {seed}")
