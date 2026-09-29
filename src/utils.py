import os
import random

import numpy as np
import torch


def seed_everything(seed):
    # Set before CUDA initialization; keep seeded runs deterministic and in FP32.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.fp32_precision = "ieee"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
