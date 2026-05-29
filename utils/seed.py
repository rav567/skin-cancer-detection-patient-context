"""
Seed setting utilities for reproducible PyTorch training across CPU, GPU, and data loaders.

"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds (Python, NumPy, PyTorch) and optionally enable deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Seeds set to {seed} (deterministic={deterministic})")


def seed_dataloader_worker(worker_id: int) -> None:
    """Worker initialisation function for DataLoader to ensure each worker gets a unique deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)