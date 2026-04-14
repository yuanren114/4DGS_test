"""Dataset for preprocessed RGB-only dynamic video runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io import list_images, load_json, read_image


class DynamicVideoDataset(Dataset):
    """Load frames, pseudo-depth, masks, cameras, and 3D track artifacts."""

    def __init__(self, preprocess_dir: Path) -> None:
        self.preprocess_dir = Path(preprocess_dir)
        self.frame_paths = list_images(self.preprocess_dir / "frames")
        self.depth_paths = sorted((self.preprocess_dir / "depth").glob("*.npy"))
        self.mask_paths = sorted((self.preprocess_dir / "masks").glob("*.npy"))
        self.cameras = load_json(self.preprocess_dir / "camera" / "cameras.json")
        self.tracks = np.load(self.preprocess_dir / "tracks" / "tracks.npz")
        if not self.frame_paths:
            raise RuntimeError(f"No frames found under {self.preprocess_dir / 'frames'}")

    def __len__(self) -> int:
        """Return number of frames."""

        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one frame sample."""

        image = torch.from_numpy(read_image(self.frame_paths[idx])).permute(2, 0, 1).float()
        depth = torch.from_numpy(np.load(self.depth_paths[idx])).float()[None]
        mask = torch.from_numpy(np.load(self.mask_paths[idx])).float()[None]
        cam = self.cameras["frames"][idx]
        return {
            "time_index": torch.tensor(idx, dtype=torch.long),
            "image": image,
            "depth": depth,
            "mask": mask,
            "intrinsics": torch.tensor(cam["intrinsics"], dtype=torch.float32),
            "world_to_camera": torch.tensor(cam["world_to_camera"], dtype=torch.float32),
            "track_uv": torch.from_numpy(self.tracks["uv"][idx]).float(),
            "track_xyz": torch.from_numpy(self.tracks["xyz"][idx]).float(),
            "track_confidence": torch.from_numpy(self.tracks["confidence"][idx]).float(),
        }

    @property
    def image_size(self) -> tuple[int, int]:
        """Return image height and width."""

        image = read_image(self.frame_paths[0])
        return image.shape[0], image.shape[1]

    @property
    def num_tracks(self) -> int:
        """Return number of pseudo tracks."""

        return int(self.tracks["uv"].shape[1])
