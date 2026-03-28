"""Utility helpers for SymNet."""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available! Using GPU: {device_name}")
        return torch.device("cuda")
    print("CUDA not detected. Falling back to CPU.")
    return torch.device("cpu")


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (3,5,5) numpy obs to (1,3,5,5) float tensor."""
    return torch.from_numpy(obs).float().unsqueeze(0).to(device)


def make_checkpoint_dir(base: str = "checkpoints") -> str:
    """Create checkpoint directory if it does not exist."""
    os.makedirs(base, exist_ok=True)
    return base
