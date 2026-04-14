"""Reconstruction and regularization losses."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.config import TrainConfig
from src.utils.camera import project_points


def gradient_smoothness(depth: torch.Tensor) -> torch.Tensor:
    """Total variation style depth smoothness."""

    dx = torch.abs(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    dy = torch.abs(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    return dx + dy


def simple_ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Small differentiable SSIM-style loss used as a D-SSIM substitute."""

    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x[None], 3, 1, 1)
    mu_y = F.avg_pool2d(y[None], 3, 1, 1)
    sig_x = F.avg_pool2d(x[None] * x[None], 3, 1, 1) - mu_x.square()
    sig_y = F.avg_pool2d(y[None] * y[None], 3, 1, 1) - mu_y.square()
    sig_xy = F.avg_pool2d(x[None] * y[None], 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x.square() + mu_y.square() + c1) * (sig_x + sig_y + c2) + 1.0e-8)
    return ((1.0 - ssim) * 0.5).mean()


def arap_loss(model) -> torch.Tensor:
    """Encourage neighboring node distances to stay stable over time."""

    if model.num_frames < 2:
        return model.node_positions.sum() * 0.0
    base = model.node_positions
    d0 = torch.cdist(base, base)
    moved = base[None] + model.node_offsets
    losses = []
    for t in range(model.num_frames):
        dt = torch.cdist(moved[t], moved[t])
        losses.append(torch.abs(dt - d0).mean())
    return torch.stack(losses).mean()


def track_projection_loss(rendered: Dict[str, torch.Tensor], sample: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Weakly align rendered Gaussian projections to preprocessed 2D tracks."""

    means = rendered["means_t"]
    uv, _ = project_points(means, sample["intrinsics"], sample["world_to_camera"])
    target = sample["track_uv"]
    if target.numel() == 0:
        return means.sum() * 0.0
    n = min(uv.shape[0], target.shape[0])
    conf = sample["track_confidence"][:n].to(uv.device)
    return (torch.linalg.norm(uv[:n] - target[:n].to(uv.device), dim=-1) * conf).mean() / max(rendered["rgb"].shape[-2:])


def compute_losses(rendered: Dict[str, torch.Tensor], sample: Dict[str, torch.Tensor], model, cfg: TrainConfig) -> Dict[str, torch.Tensor]:
    """Compute all major training losses."""

    image = sample["image"].to(rendered["rgb"].device)
    depth = sample["depth"].to(rendered["rgb"].device)
    mask = sample["mask"].to(rendered["rgb"].device)
    rgb_l1 = F.l1_loss(rendered["rgb"], image)
    dssim = simple_ssim_loss(rendered["rgb"], image)
    mask_loss = F.mse_loss(rendered["mask"], mask)
    depth_loss = F.mse_loss(rendered["depth"] * mask, depth * mask)
    smooth = gradient_smoothness(rendered["depth"])
    track = track_projection_loss(rendered, sample)
    arap = arap_loss(model)
    total = (
        cfg.rgb_weight * rgb_l1
        + cfg.dssim_weight * dssim
        + cfg.mask_weight * mask_loss
        + cfg.depth_weight * depth_loss
        + cfg.depth_smooth_weight * smooth
        + cfg.track_weight * track
        + cfg.arap_weight * arap
    )
    return {
        "total": total,
        "rgb_l1": rgb_l1,
        "dssim": dssim,
        "mask": mask_loss,
        "depth": depth_loss,
        "depth_smooth": smooth,
        "track": track,
        "arap": arap,
    }
