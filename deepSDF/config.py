import torch
import random
import numpy as np


class DataConfiguration:
    RAW_DATA_PATH = "deepSDF/data/raw"
    SAVE_DATA_PATH = "deepSDF/data/preprocessed"

    N_TOTAL_SAMPLING = 32**3
    N_SURFACE_SAMPLING_RATIO = 0.3
    N_BBOX_SAMPLING_RATIO = 0.2
    N_VOLUME_SAMPLING_RATIO = 0.5


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
