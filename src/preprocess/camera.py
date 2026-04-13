"""Camera estimation wrappers for RGB-only input."""

from __future__ import annotations

from pathlib import Path
from typing import List
import shutil
import subprocess

import numpy as np
import torch

from src.utils.camera import default_intrinsics, look_at
from src.utils.io import read_image, save_json


class CameraEstimator:
    """Estimate camera metadata needed by the dynamic Gaussian pipeline."""

    def __init__(self, method: str = "colmap_or_simple_vo") -> None:
        self.method = method

    def estimate(self, frame_paths: List[Path], output_dir: Path) -> Path:
        """Estimate cameras and save a portable camera JSON file."""

        output_dir.mkdir(parents=True, exist_ok=True)
        if self.method.startswith("colmap") and shutil.which("colmap") is not None:
            self._write_colmap_instructions(output_dir)
        cameras = self._simple_orbit_proxy(frame_paths)
        path = output_dir / "cameras.json"
        save_json(path, cameras)
        return path

    def _write_colmap_instructions(self, output_dir: Path) -> None:
        """Record that COLMAP is available but not run inside the baseline."""

        note = (
            "COLMAP executable was found. This baseline writes simple proxy cameras by default. "
            "For production runs, replace preprocess/camera/cameras.json with COLMAP poses "
            "converted to the same schema."
        )
        (output_dir / "COLMAP_AVAILABLE.txt").write_text(note, encoding="utf-8")

    def _simple_orbit_proxy(self, frame_paths: List[Path]) -> dict:
        """Create approximate object-centric cameras from RGB frame count and size."""

        sample = read_image(frame_paths[0])
        height, width = sample.shape[:2]
        k = default_intrinsics(width, height).numpy().tolist()
        frames = []
        total = max(1, len(frame_paths))
        for idx, path in enumerate(frame_paths):
            angle = (idx - (total - 1) * 0.5) * 0.04
            eye = torch.tensor([0.25 * np.sin(angle), 0.0, -3.0 + 0.05 * np.cos(angle)], dtype=torch.float32)
            pose = look_at(eye, torch.tensor([0.0, 0.0, 1.2], dtype=torch.float32))
            frames.append(
                {
                    "frame": path.name,
                    "intrinsics": k,
                    "world_to_camera": pose.numpy().tolist(),
                    "camera_to_world": torch.inverse(pose).numpy().tolist(),
                    "source": "simple_orbit_proxy_from_rgb_only",
                }
            )
        return {"width": width, "height": height, "frames": frames}


def estimate_cameras(frame_paths: List[Path], output_dir: Path, method: str) -> Path:
    """Estimate and save camera parameters."""

    return CameraEstimator(method).estimate(frame_paths, output_dir)
