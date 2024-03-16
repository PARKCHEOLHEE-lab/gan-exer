import torch
import random
import numpy as np


class DataConfiguration:
    N_TOTAL_SAMPING = 32**3

    RAW_DATA_PATH = "deepSDF/data/raw"
    SAVE_DATA_PATH = "deepSDF/data/preprocessed"


class ModelConfiguration:
    BATCH_SIZE = 32
    LATENT_SIZE = 128


class Configuration(DataConfiguration, ModelConfiguration):
    """Configuration related to the DeepSDF model"""

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
