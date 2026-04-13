"""Training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import random

import numpy as np
import torch

from src.config import AppConfig
from src.datasets.dynamic_video import DynamicVideoDataset
from src.losses.reconstruction import compute_losses
from src.models.dynamic_gaussians import DynamicGaussianModel
from src.utils.logging import RunLogger
from src.utils.visualization import save_tensor_image, save_depth


def resolve_device(device: str) -> torch.device:
    """Resolve auto/cpu/cuda device strings."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_sample(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor sample fields to device."""

    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.items()}


def train_model(config: AppConfig, run_dir: Path, logger: RunLogger) -> Path:
    """Train the dynamic Gaussian model and return the final checkpoint path."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.train.device)
    dataset = DynamicVideoDataset(run_dir / "preprocess")
    height, width = dataset.image_size
    model = DynamicGaussianModel(config.model, len(dataset), height, width).to(device)

    all_tracks = torch.from_numpy(dataset.tracks["xyz"]).float().to(device)
    all_images = torch.stack([dataset[i]["image"] for i in range(len(dataset))], dim=0).to(device)
    model.initialize_from_tracks(all_tracks, all_images)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    ckpt_dir = run_dir / "checkpoints"
    vis_dir = run_dir / "visualizations"
    debug_dir = run_dir / "debug"

    logger.log(f"train.device: {device}")
    logger.log(f"train.dataset: frames={len(dataset)}, image_size={height}x{width}, tracks={dataset.num_tracks}")
    logger.log(f"train.model: gaussians={config.model.num_gaussians}, nodes={config.model.num_nodes}")

    final_ckpt = ckpt_dir / "final.pt"
    for step in range(1, config.train.iterations + 1):
        idx = (step - 1) % len(dataset)
        sample = move_sample(dataset[idx], device)
        rendered = model(sample)
        losses = compute_losses(rendered, sample, model, config.train)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        if step == 1:
            logger.log(f"debug.tensor.image: {tuple(sample['image'].shape)}")
            logger.log(f"debug.tensor.render_rgb: {tuple(rendered['rgb'].shape)}")
            logger.log(f"debug.tensor.means_t: {tuple(rendered['means_t'].shape)}")
            save_tensor_image(debug_dir / "step_0001_render.png", rendered["rgb"])
            save_tensor_image(debug_dir / "step_0001_target.png", sample["image"])
            save_depth(debug_dir / "step_0001_depth.png", rendered["depth"][0])

        if step % config.train.log_every == 0 or step == 1:
            scalars = {k: float(v.detach().cpu()) for k, v in losses.items()}
            logger.log_dict(f"train.step {step:05d}", scalars)

        if step % config.train.save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "config": config}, ckpt_path)
            save_tensor_image(vis_dir / f"render_step_{step:06d}.png", rendered["rgb"])

    torch.save({"model": model.state_dict(), "config": config}, final_ckpt)
    logger.log(f"train.checkpoint: {final_ckpt}")
    return final_ckpt
