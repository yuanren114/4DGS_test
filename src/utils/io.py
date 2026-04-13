"""I/O helpers for images, arrays, metadata, and run folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
from PIL import Image


def read_image(path: Path) -> np.ndarray:
    """Read an RGB image as float32 in [0, 1]."""

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an RGB or grayscale image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def write_depth_png(path: Path, depth: np.ndarray) -> None:
    """Write a normalized depth preview image."""

    d = np.asarray(depth, dtype=np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1.0e-8)
    write_image(path, d)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save JSON with readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_images(path: Path) -> List[Path]:
    """Return sorted image paths."""

    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
