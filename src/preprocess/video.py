"""RGB video frame extraction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from src.utils.io import write_image


def _resize(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize an image so the longest side equals image_size."""

    h, w = image.shape[:2]
    scale = float(image_size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))


def extract_frames(
    input_video: str,
    output_dir: Path,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    image_size: int = 256,
) -> List[Path]:
    """Extract RGB frames from a video file or copy frames from an image directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_video)
    frames: List[np.ndarray] = []

    if input_path.is_dir():
        image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for idx, path in enumerate(image_paths):
            if idx % frame_stride != 0:
                continue
            image = np.asarray(Image.open(path).convert("RGB"))
            frames.append(_resize(image, image_size))
            if max_frames is not None and len(frames) >= max_frames:
                break
    elif input_path.exists():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to extract frames from video files.") from exc
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_video}")
        raw_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if raw_idx % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(_resize(frame_rgb, image_size))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            raw_idx += 1
        cap.release()
    else:
        raise FileNotFoundError(f"Input video or frame directory does not exist: {input_video}")

    saved: List[Path] = []
    for idx, frame in enumerate(frames):
        path = output_dir / f"{idx:05d}.png"
        write_image(path, frame)
        saved.append(path)
    return saved


def create_synthetic_frames(output_dir: Path, num_frames: int = 6, image_size: int = 96) -> List[Path]:
    """Create a tiny RGB-only synthetic sequence for quick wiring tests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    for t in range(num_frames):
        image = np.ones((image_size, image_size, 3), dtype=np.float32)
        image[..., 0] *= 0.92
        image[..., 1] *= 0.96
        image[..., 2] *= 1.00
        cx = image_size * (0.35 + 0.25 * t / max(1, num_frames - 1))
        cy = image_size * (0.5 + 0.08 * np.sin(t))
        radius = image_size * 0.18
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = np.array([0.1, 0.45, 0.95], dtype=np.float32)
        image[mask & (yy < cy)] = np.array([0.95, 0.3, 0.15], dtype=np.float32)
        path = output_dir / f"{t:05d}.png"
        write_image(path, image)
        saved.append(path)
    return saved
