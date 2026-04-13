"""Foreground mask estimation from RGB-only frames."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.utils.io import read_image, write_image


def estimate_masks(frame_paths: List[Path], output_dir: Path, method: str = "saliency_fallback") -> List[Path]:
    """Estimate foreground masks with a deterministic RGB saliency fallback."""

    output_dir.mkdir(parents=True, exist_ok=True)
    images = [read_image(p) for p in frame_paths]
    median_bg = np.median(np.stack(images, axis=0), axis=0)
    saved: List[Path] = []
    for path, image in zip(frame_paths, images):
        diff = np.linalg.norm(image - median_bg, axis=-1)
        color_saliency = np.linalg.norm(image - image.reshape(-1, 3).mean(axis=0), axis=-1)
        score = 0.6 * diff + 0.4 * color_saliency
        threshold = np.percentile(score, 65.0)
        mask = (score >= threshold).astype(np.float32)
        npy_path = output_dir / f"{path.stem}.npy"
        np.save(npy_path, mask)
        write_image(output_dir / f"{path.stem}.png", mask)
        saved.append(npy_path)
    return saved
