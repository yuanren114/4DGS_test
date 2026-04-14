"""End-to-end preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from src.config import AppConfig
from src.preprocess.camera import estimate_cameras
from src.preprocess.depth import estimate_depth_sequence
from src.preprocess.masks import StopAfterMaskInit, estimate_masks
from src.preprocess.track_backends import get_3d_tracks
from src.preprocess.video import create_synthetic_frames, extract_frames
from src.utils.io import save_json
from src.utils.logging import RunLogger


def run_preprocessing(config: AppConfig, run_dir: Path, logger: RunLogger) -> Dict[str, str]:
    """Run RGB-only preprocessing and return artifact paths."""

    pp_dir = run_dir / "preprocess"
    frames_dir = pp_dir / "frames"
    depth_dir = pp_dir / "depth"
    camera_dir = pp_dir / "camera"
    mask_dir = pp_dir / "masks"
    track_dir = pp_dir / "tracks"
    metadata_dir = pp_dir / "metadata"

    if config.quick_test and not config.preprocess.input_video:
        frame_paths = create_synthetic_frames(frames_dir, num_frames=6, image_size=96)
    else:
        frame_paths = extract_frames(
            config.preprocess.input_video,
            frames_dir,
            frame_stride=config.preprocess.frame_stride,
            max_frames=config.preprocess.max_frames,
            image_size=config.preprocess.image_size,
        )
    logger.log(f"preprocess.frames: {len(frame_paths)} frames -> {frames_dir}")

    try:
        mask_paths = estimate_masks(
            frame_paths=frame_paths,
            output_dir=mask_dir,
            config=config.preprocess,
            preview_dir=pp_dir / "masks_preview",
            candidates_dir=pp_dir / "masks_candidates",
            debug_dir=pp_dir / "masks_debug",
            device=config.train.device if config.train.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu",
        )
    except StopAfterMaskInit as exc:
        logger.log("preprocess.masks: stopped after first-frame initialization")
        for name, path in exc.debug_paths.items():
            logger.log(f"preprocess.masks.debug.{name}: {path}")
        metadata = {
            "frames_dir": str(frames_dir),
            "masks_dir": str(mask_dir),
            "mask_debug_dir": str(pp_dir / "masks_debug"),
            "stopped_after_init": True,
            "rgb_only": True,
        }
        save_json(metadata_dir / "preprocess_manifest.json", metadata)
        return metadata
    logger.log(f"preprocess.masks: {config.preprocess.mask_method}, {len(mask_paths)} masks -> {mask_dir}")

    depth_paths = estimate_depth_sequence(frame_paths, depth_dir, config.preprocess.depth_method)
    logger.log(f"preprocess.depth: {len(depth_paths)} pseudo-depth maps -> {depth_dir}")

    camera_path = estimate_cameras(frame_paths, camera_dir, config.preprocess.camera_method)
    logger.log(f"preprocess.camera: camera metadata -> {camera_path}")

    tracks_path = get_3d_tracks(
        frame_paths=frame_paths,
        depth_paths=depth_paths,
        camera_json=camera_path,
        mask_paths=mask_paths,
        output_dir=track_dir,
        config=config.preprocess,
        device=config.train.device if config.train.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu",
    )
    logger.log(f"preprocess.tracks: {config.preprocess.track_method} -> {tracks_path}")

    metadata = {
        "frames_dir": str(frames_dir),
        "depth_dir": str(depth_dir),
        "camera_json": str(camera_path),
        "masks_dir": str(mask_dir),
        "mask_debug_dir": str(pp_dir / "masks_debug"),
        "tracks_npz": str(tracks_path),
        "rgb_only": True,
    }
    save_json(metadata_dir / "preprocess_manifest.json", metadata)
    return metadata
