import os
import torch
import random
import numpy as np


class ModelConfig:
    """Configuration related to the GAN models"""

    DATA_PATH = os.path.abspath(os.path.join(__file__, "../", "data", "binpy"))

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

    EPOCHS = 1000
    LOG_INTERVAL = int(EPOCHS / 10)
    LEARNING_RATE = 0.0001
    BETAS = (0.5, 0.999)
    BATCH_SIZE = 100
    BATCH_SIZE_TO_EVALUATE = 5

    XAVIER = "xavier"
    HE = "he"
