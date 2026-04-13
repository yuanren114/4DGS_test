"""2D and pseudo-3D track initialization from RGB, depth, and cameras."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from src.utils.camera import unproject_pixels
from src.utils.io import load_json, read_image


def build_grid_tracks(
    frame_paths: List[Path],
    depth_paths: List[Path],
    camera_json: Path,
    mask_paths: List[Path],
    output_dir: Path,
    grid_size: int = 24,
    confidence_threshold: float = 0.5,
) -> Path:
    """Build practical pseudo-tracks.

    This is an engineering substitute for BootsTAP + AnchorTAP3D. It samples
    foreground grid points, applies optional optical flow if OpenCV is available,
    and unprojects every observation with pseudo-depth and estimated cameras.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_json(camera_json)
    first = read_image(frame_paths[0])
    h, w = first.shape[:2]
    mask0 = np.load(mask_paths[0])
    ys = np.linspace(0, h - 1, grid_size)
    xs = np.linspace(0, w - 1, grid_size)
    points = []
    for y in ys:
        for x in xs:
            if mask0[int(round(y)), int(round(x))] > 0.0:
                points.append([x, y])
    if not points:
        points = [[x, y] for y in ys for x in xs]
    uv0 = np.asarray(points, dtype=np.float32)
    uv_tracks = np.repeat(uv0[None, :, :], len(frame_paths), axis=0)
    confidence = np.ones((len(frame_paths), uv0.shape[0]), dtype=np.float32) * 0.75

    try:
        import cv2

        prev_gray = (first.mean(axis=-1) * 255).astype(np.uint8)
        prev_pts = uv0.reshape(-1, 1, 2)
        for t in range(1, len(frame_paths)):
            image = read_image(frame_paths[t])
            gray = (image.mean(axis=-1) * 255).astype(np.uint8)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            if next_pts is not None and status is not None:
                uv_tracks[t] = next_pts.reshape(-1, 2)
                confidence[t] = status.reshape(-1).astype(np.float32)
                prev_pts = next_pts
            prev_gray = gray
    except Exception:
        pass

    xyz_tracks = []
    for t in range(len(frame_paths)):
        cam = cameras["frames"][t]
        intr = torch.tensor(cam["intrinsics"], dtype=torch.float32)
        c2w = torch.tensor(cam["camera_to_world"], dtype=torch.float32)
        depth = np.load(depth_paths[t])
        uv = np.clip(uv_tracks[t], [0, 0], [w - 1, h - 1])
        d = depth[np.round(uv[:, 1]).astype(int), np.round(uv[:, 0]).astype(int)]
        xyz = unproject_pixels(torch.tensor(uv, dtype=torch.float32), torch.tensor(d, dtype=torch.float32), intr, c2w)
        xyz_tracks.append(xyz.numpy())
    xyz_tracks_np = np.stack(xyz_tracks, axis=0).astype(np.float32)

    path = output_dir / "tracks.npz"
    np.savez(
        path,
        uv=uv_tracks.astype(np.float32),
        xyz=xyz_tracks_np,
        confidence=confidence,
        confidence_threshold=np.array([confidence_threshold], dtype=np.float32),
        source=np.array(["grid_optical_flow_anchor_substitute"]),
    )
    return path
