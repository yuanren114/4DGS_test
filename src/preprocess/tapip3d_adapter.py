"""TAPIP3D adapter using the official TAPIP3D inference code."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import subprocess
import sys

import numpy as np

from src.utils.io import load_json, read_image


class TAPIP3DAdapter:
    """Run official TAPIP3D inference on prepared RGB/depth/camera arrays."""

    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        resolution_factor: int = 1,
        num_iters: int = 6,
        support_grid_size: int = 16,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.resolution_factor = resolution_factor
        self.num_iters = num_iters
        self.support_grid_size = support_grid_size
        if not self.repo_path.exists():
            raise FileNotFoundError(f"TAPIP3D repository not found: {self.repo_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"TAPIP3D checkpoint not found: {self.checkpoint_path}")

    def track(
        self,
        frame_paths: List[Path],
        depth_paths: List[Path],
        camera_json: Path,
        query_points_txyz: np.ndarray,
        output_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray, Path]:
        """Run TAPIP3D and return world-space coordinates and visibility."""

        output_dir.mkdir(parents=True, exist_ok=True)
        input_npz = self._write_input_npz(frame_paths, depth_paths, camera_json, query_points_txyz, output_dir)
        before = set(output_dir.rglob("*.result.npz"))
        command = [
            sys.executable,
            "inference.py",
            "--input_path",
            str(input_npz.resolve()),
            "--checkpoint",
            str(self.checkpoint_path.resolve()),
            "--device",
            self.device,
            "--resolution_factor",
            str(self.resolution_factor),
            "--num_iters",
            str(self.num_iters),
            "--support_grid_size",
            str(self.support_grid_size),
            "--output_dir",
            str(output_dir.resolve()),
        ]
        subprocess.run(command, cwd=self.repo_path, check=True)
        after = set(output_dir.rglob("*.result.npz"))
        new_results = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
        if not new_results:
            new_results = sorted(output_dir.rglob("*.result.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not new_results:
            raise RuntimeError("TAPIP3D completed without producing a .result.npz file.")
        result_path = new_results[0]
        data = np.load(result_path)
        return data["coords"].astype(np.float32), data["visibs"].astype(np.float32), result_path

    def _write_input_npz(
        self,
        frame_paths: List[Path],
        depth_paths: List[Path],
        camera_json: Path,
        query_points_txyz: np.ndarray,
        output_dir: Path,
    ) -> Path:
        """Write a TAPIP3D-compatible input archive."""

        cameras = load_json(camera_json)
        video = np.stack([(read_image(p) * 255.0).astype(np.uint8) for p in frame_paths], axis=0)
        depths = np.stack([np.load(p).astype(np.float32) for p in depth_paths], axis=0)
        intrinsics = np.stack([np.asarray(frame["intrinsics"], dtype=np.float32) for frame in cameras["frames"]], axis=0)
        extrinsics = np.stack([np.asarray(frame["world_to_camera"], dtype=np.float32) for frame in cameras["frames"]], axis=0)
        path = output_dir / "tapip3d_input.npz"
        np.savez(
            path,
            video=video,
            depths=depths.astype(np.float32),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_point=query_points_txyz.astype(np.float32),
        )
        return path
