"""BootsTAPIR adapter using Google DeepMind TAP-Net's PyTorch model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import importlib
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.io import read_image


class BootsTAPIRAdapter:
    """Run the official TAP-Net PyTorch BootsTAPIR model when available."""

    def __init__(self, repo_path: str, checkpoint_path: str, device: str = "cpu") -> None:
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        if not self.repo_path.exists():
            raise FileNotFoundError(f"TAP-Net repository not found: {self.repo_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"BootsTAPIR checkpoint not found: {self.checkpoint_path}")

    def track(self, frame_paths: List[Path], query_points_tyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return 2D tracks and confidence from BootsTAPIR.

        Args:
            frame_paths: RGB frame paths.
            query_points_tyx: Query points shaped `[N, 3]` in `(t, y, x)`.

        Returns:
            `uv_tracks` shaped `[T, N, 2]` in `(x, y)` and confidence shaped
            `[T, N]`. The confidence follows the BootsTAPIR demo convention:
            `(1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))`.
        """

        tapir_model = self._import_torch_tapir_without_top_level_init()

        frames_np = np.stack([(read_image(p) * 255.0).astype(np.uint8) for p in frame_paths], axis=0)
        frames = torch.from_numpy(frames_np).to(self.device).float()
        frames = frames / 255.0 * 2.0 - 1.0
        query_points = torch.from_numpy(query_points_tyx.astype(np.float32)).to(self.device)

        model = tapir_model.TAPIR(pyramid_level=1)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model = model.to(self.device).eval()

        with torch.no_grad():
            outputs = model(frames[None], query_points[None])
            tracks_yx = outputs["tracks"][0]
            occlusion = outputs["occlusion"][0]
            expected_dist = outputs["expected_dist"][0]
            confidence = (1.0 - torch.sigmoid(occlusion)) * (1.0 - torch.sigmoid(expected_dist))

        tracks_yx_np = tracks_yx.detach().cpu().numpy().astype(np.float32)
        uv_tracks = tracks_yx_np[..., [1, 0]]
        return uv_tracks, confidence.detach().cpu().numpy().astype(np.float32)

    def _import_torch_tapir_without_top_level_init(self):
        """Import `tapnet.torch.tapir_model` without TensorFlow-heavy package init."""

        root = self.repo_path.resolve() / "tapnet"
        tapnet_pkg = types.ModuleType("tapnet")
        tapnet_pkg.__path__ = [str(root)]
        torch_pkg = types.ModuleType("tapnet.torch")
        torch_pkg.__path__ = [str(root / "torch")]
        sys.modules.setdefault("tapnet", tapnet_pkg)
        sys.modules.setdefault("tapnet.torch", torch_pkg)
        return importlib.import_module("tapnet.torch.tapir_model")
