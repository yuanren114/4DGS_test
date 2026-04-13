"""Depth estimation wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.io import read_image, write_depth_png


class DepthEstimator:
    """Estimate pseudo-depth from RGB frames.

    The preferred real-data path is Depth Anything v2 via Hugging Face
    transformers. A deterministic luminance/saliency fallback keeps quick tests
    runnable without downloading model weights.
    """

    def __init__(self, method: str = "depth_anything_v2", device: str = "cpu") -> None:
        self.method = method
        self.device = device
        self.processor = None
        self.model = None
        if method.startswith("depth_anything"):
            self._try_load_depth_anything()

    def _try_load_depth_anything(self) -> None:
        """Try to load Depth Anything v2 without making it mandatory."""

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()
        except Exception:
            self.processor = None
            self.model = None

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate one normalized pseudo-depth map."""

        if self.model is not None and self.processor is not None:
            from PIL import Image

            pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                pred = self.model(**inputs).predicted_depth[:, None]
                pred = F.interpolate(pred, size=image.shape[:2], mode="bicubic", align_corners=False)
            depth = pred[0, 0].detach().cpu().numpy().astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1.0e-8)
            return 0.5 + 1.5 * depth

        gray = image.mean(axis=-1).astype(np.float32)
        yy = np.linspace(0.0, 1.0, image.shape[0], dtype=np.float32)[:, None]
        saliency = np.abs(gray - np.median(gray))
        depth = 1.5 + 0.6 * yy - 0.7 * saliency
        depth = np.clip(depth, 0.25, 3.0)
        return depth.astype(np.float32)


def estimate_depth_sequence(frame_paths: List[Path], output_dir: Path, method: str, device: str = "cpu") -> List[Path]:
    """Estimate and save pseudo-depth arrays plus preview images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = DepthEstimator(method=method, device=device)
    saved: List[Path] = []
    for frame_path in frame_paths:
        image = read_image(frame_path)
        depth = estimator.estimate(image)
        npy_path = output_dir / f"{frame_path.stem}.npy"
        np.save(npy_path, depth)
        write_depth_png(output_dir / f"{frame_path.stem}.png", depth)
        saved.append(npy_path)
    return saved
