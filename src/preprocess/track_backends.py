"""Clean interface for 3D track generation backends."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from src.config import PreprocessConfig
from src.preprocess.boots_tap_adapter import BootsTAPIRAdapter
from src.preprocess.proxy_3d_tracks import build_proxy_3d_tracks
from src.preprocess.tapip3d_adapter import TAPIP3DAdapter
from src.utils.camera import unproject_pixels
from src.utils.io import load_json, read_image


def get_3d_tracks(
    frame_paths: List[Path],
    depth_paths: List[Path],
    camera_json: Path,
    mask_paths: List[Path],
    output_dir: Path,
    config: PreprocessConfig,
    device: str = "cpu",
) -> Path:
    """Generate 3D tracks for downstream Gaussian initialization.

    Available backends:
    - `proxy_grid_lk`: runnable baseline with grid sampling, LK optical flow,
      pseudo-depth, and camera unprojection.
    - `bootstap_tapip3d_components`: uses official BootsTAPIR and TAPIP3D
      repositories/checkpoints when installed. This integrates the named
      components but is still not the full paper AnchorTAP3D anchor-window
      algorithm unless that missing 4DGS360 code is supplied.
    """

    if config.track_method == "proxy_grid_lk":
        return build_proxy_3d_tracks(frame_paths, depth_paths, camera_json, mask_paths, output_dir)
    if config.track_method == "bootstap_tapip3d_components":
        return _run_bootstap_tapip3d_components(frame_paths, depth_paths, camera_json, mask_paths, output_dir, config, device)
    raise ValueError(f"Unknown track_method: {config.track_method}")


def _run_bootstap_tapip3d_components(
    frame_paths: List[Path],
    depth_paths: List[Path],
    camera_json: Path,
    mask_paths: List[Path],
    output_dir: Path,
    config: PreprocessConfig,
    device: str,
) -> Path:
    """Run official BootsTAPIR and TAPIP3D components behind one interface."""

    output_dir.mkdir(parents=True, exist_ok=True)
    query_tyx, query_txyz = _make_first_frame_queries(frame_paths, depth_paths, camera_json, mask_paths)

    bootstap = BootsTAPIRAdapter(config.bootstap_repo, config.bootstap_checkpoint, device=device)
    uv_tracks, bootstap_conf = bootstap.track(frame_paths, query_tyx)
    np.savez(output_dir / "bootstapir_2d_tracks.npz", uv=uv_tracks, confidence=bootstap_conf, query_tyx=query_tyx)

    tapip3d = TAPIP3DAdapter(
        repo_path=config.tapip3d_repo,
        checkpoint_path=config.tapip3d_checkpoint,
        device=device,
        resolution_factor=config.tapip3d_resolution_factor,
        num_iters=config.tapip3d_num_iters,
        support_grid_size=config.tapip3d_support_grid_size,
    )
    xyz_tracks, tapip_vis, result_path = tapip3d.track(frame_paths, depth_paths, camera_json, query_txyz, output_dir / "tapip3d")

    n = min(uv_tracks.shape[1], xyz_tracks.shape[1])
    path = output_dir / "tracks.npz"
    np.savez(
        path,
        uv=uv_tracks[:, :n].astype(np.float32),
        xyz=xyz_tracks[:, :n].astype(np.float32),
        confidence=(bootstap_conf[:, :n] * tapip_vis[:, :n]).astype(np.float32),
        confidence_threshold=np.array([0.5], dtype=np.float32),
        source=np.array(["official_bootstapir_plus_official_tapip3d_components"]),
        backend=np.array(["bootstap_tapip3d_components"]),
        paper_faithful=np.array([False]),
        tapip3d_result=str(result_path),
    )
    return path


def _make_first_frame_queries(
    frame_paths: List[Path],
    depth_paths: List[Path],
    camera_json: Path,
    mask_paths: List[Path],
    grid_size: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Create first-frame query points for BootsTAPIR and TAPIP3D."""

    first = read_image(frame_paths[0])
    height, width = first.shape[:2]
    mask0 = np.load(mask_paths[0])
    ys = np.linspace(0, height - 1, grid_size)
    xs = np.linspace(0, width - 1, grid_size)
    xy = []
    for y in ys:
        for x in xs:
            if mask0[int(round(y)), int(round(x))] > 0.0:
                xy.append([x, y])
    if not xy:
        xy = [[x, y] for y in ys for x in xs]
    uv = np.asarray(xy, dtype=np.float32)
    query_tyx = np.concatenate([np.zeros((len(uv), 1), dtype=np.float32), uv[:, [1, 0]]], axis=-1)

    cameras = load_json(camera_json)
    intr = torch.tensor(cameras["frames"][0]["intrinsics"], dtype=torch.float32)
    c2w = torch.tensor(cameras["frames"][0]["camera_to_world"], dtype=torch.float32)
    depth0 = np.load(depth_paths[0])
    uv_clip = np.clip(uv, [0, 0], [width - 1, height - 1])
    d = depth0[np.round(uv_clip[:, 1]).astype(int), np.round(uv_clip[:, 0]).astype(int)]
    xyz = unproject_pixels(torch.tensor(uv_clip, dtype=torch.float32), torch.tensor(d, dtype=torch.float32), intr, c2w).numpy()
    query_txyz = np.concatenate([np.zeros((len(xyz), 1), dtype=np.float32), xyz.astype(np.float32)], axis=-1)
    return query_tyx.astype(np.float32), query_txyz.astype(np.float32)
