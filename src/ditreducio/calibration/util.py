from __future__ import annotations

import os
import random

import numpy as np
import torch


def threshold_q(data: np.ndarray, ratio: float = 0.5) -> float:
    return float(np.percentile(data, (1.0 - ratio) * 100.0))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
