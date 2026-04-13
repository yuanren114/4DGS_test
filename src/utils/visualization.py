"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.io import write_depth_png, write_image


def save_tensor_image(path: Path, tensor: torch.Tensor) -> None:
    """Save a CHW or HWC tensor image."""

    arr = tensor.detach().cpu().float()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.permute(1, 2, 0)
    arr_np = arr.numpy()
    if arr_np.shape[-1] == 1:
        arr_np = arr_np[..., 0]
    write_image(path, arr_np)


def save_depth(path: Path, tensor: torch.Tensor) -> None:
    """Save a depth tensor preview."""

    write_depth_png(path, tensor.detach().cpu().numpy())


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Concatenate two images for quick inspection."""

    return np.concatenate([left, right], axis=1)
