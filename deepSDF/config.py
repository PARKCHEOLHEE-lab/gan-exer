import torch
import random
import numpy as np


class DataConfiguration:
    RAW_DATA_PATH = "deepSDF/data/raw-skyscrapers"
    SAVE_DATA_PATH = "deepSDF/data/preprocessed-skyscrapers"

    N_TOTAL_SAMPLING = 64**3
    N_SURFACE_SAMPLING_RATIO = 0.3
    N_BBOX_SAMPLING_RATIO = 0.2
    N_VOLUME_SAMPLING_RATIO = 0.5

    N_SURFACE_SAMPLING = int(N_TOTAL_SAMPLING * N_SURFACE_SAMPLING_RATIO)
    N_BBOX_SAMPLING = int(N_TOTAL_SAMPLING * N_BBOX_SAMPLING_RATIO)
    N_VOLUME_SAMPLING = int(N_TOTAL_SAMPLING * N_VOLUME_SAMPLING_RATIO)

    if (N_SURFACE_SAMPLING + N_BBOX_SAMPLING + N_VOLUME_SAMPLING) < N_TOTAL_SAMPLING:
        N_VOLUME_SAMPLING += N_TOTAL_SAMPLING - (N_SURFACE_SAMPLING + N_BBOX_SAMPLING + N_VOLUME_SAMPLING)

    assert (
        N_SURFACE_SAMPLING + N_BBOX_SAMPLING + N_VOLUME_SAMPLING
    ) == N_TOTAL_SAMPLING, "The sum of sampling `n` is not equal to `n_total_sampling`"

    RECONSTRUCT_RESOLUTION = 256


class ModelConfiguration:
    BATCH_SIZE = 64
    LATENT_SIZE = 128
    CLAMP_VALUE = 0.1

    EPOCHS = 20
    LOG_INTERVAL = 1

    LEARNING_RATE_MODEL = 0.00001
    LEARNING_RATE_LATENT = 0.001

    TRAIN_DATASET_RATIO = 0.8
    VAL_DATASET_RATIO = 0.2


class Configuration(DataConfiguration, ModelConfiguration):
    """Configuration related to the DeepSDF model"""

    def __iter__(self):
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("_"):
                yield attr, getattr(self, attr)

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    DEFAULT_SEED = 777
    SEED_SET = None

    LOG_DIR = "deepSDF/runs"

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

        Configuration.SEED_SET = seed
