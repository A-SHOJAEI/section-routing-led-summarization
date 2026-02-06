from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # For CUDA deterministic matmul paths.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_torch_reproducibility(seed: int, deterministic: bool) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not have deterministic implementations; treat as best-effort.
            pass

