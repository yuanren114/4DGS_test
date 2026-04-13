"""Inference pipeline."""

from __future__ import annotations

from pathlib import Path
import math

import torch

from src.config import AppConfig
from src.datasets.dynamic_video import DynamicVideoDataset
from src.models.dynamic_gaussians import DynamicGaussianModel
from src.pipelines.training import move_sample, resolve_device
from src.utils.camera import look_at
from src.utils.logging import RunLogger
from src.utils.visualization import save_tensor_image


def run_inference(config: AppConfig, run_dir: Path, checkpoint_path: Path, logger: RunLogger) -> None:
    """Render training-view and simple bullet-time outputs."""

    device = resolve_device(config.train.device)
    dataset = DynamicVideoDataset(run_dir / "preprocess")
    height, width = dataset.image_size
    model = DynamicGaussianModel(config.model, len(dataset), height, width).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out_dir = run_dir / "inference"

    with torch.no_grad():
        idx = min(config.inference.time_index, len(dataset) - 1)
        sample = move_sample(dataset[idx], device)
        rendered = model(sample)
        save_tensor_image(out_dir / f"train_view_t{idx:04d}.png", rendered["rgb"])

        base_sample = dict(sample)
        for view_idx in range(config.inference.num_bullet_time_views):
            angle = 2.0 * math.pi * view_idx / max(1, config.inference.num_bullet_time_views)
            eye = torch.tensor([3.0 * math.sin(angle), 0.3, -3.0 * math.cos(angle)], dtype=torch.float32, device=device)
            pose = look_at(eye.cpu(), torch.tensor([0.0, 0.0, 1.2])).to(device)
            base_sample["world_to_camera"] = pose
            rendered_bt = model(base_sample)
            save_tensor_image(out_dir / f"bullet_t{idx:04d}_view{view_idx:03d}.png", rendered_bt["rgb"])
    logger.log(f"inference.outputs: {out_dir}")
