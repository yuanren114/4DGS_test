"""Differentiable point-Gaussian renderer implemented in PyTorch."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.utils.camera import project_points


class GaussianRenderer(nn.Module):
    """Render colored isotropic Gaussian splats to RGB, mask, and depth."""

    def __init__(self, height: int, width: int, background: float = 1.0) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.background = background
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).float()
        self.register_buffer("pixel_grid", grid, persistent=False)

    def forward(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        opacity: torch.Tensor,
        scales: torch.Tensor,
        intrinsics: torch.Tensor,
        world_to_camera: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render a set of 3D Gaussians."""

        uv, z = project_points(means, intrinsics, world_to_camera)
        valid = (z > 1.0e-4) & (uv[:, 0] >= -self.width) & (uv[:, 0] <= 2 * self.width) & (uv[:, 1] >= -self.height) & (uv[:, 1] <= 2 * self.height)
        uv = uv[valid]
        z = z[valid]
        colors = colors[valid]
        opacity = opacity[valid]
        scales = scales[valid]
        if uv.numel() == 0:
            rgb = torch.full((3, self.height, self.width), self.background, device=means.device, dtype=means.dtype)
            return {"rgb": rgb, "mask": torch.zeros((1, self.height, self.width), device=means.device), "depth": torch.zeros((1, self.height, self.width), device=means.device)}

        grid = self.pixel_grid.to(means.device, means.dtype)
        diff = grid[None] - uv[:, None, None, :]
        sigma = (scales[:, None, None, 0] * max(self.height, self.width)).clamp_min(1.0)
        alpha = opacity[:, None, None] * torch.exp(-0.5 * (diff.square().sum(dim=-1) / sigma.square()))
        alpha = alpha.clamp(0.0, 0.95)
        weights = alpha / (alpha.sum(dim=0, keepdim=True) + 1.0e-6)
        accum_alpha = alpha.sum(dim=0).clamp(0.0, 1.0)
        rgb = (weights[:, None] * colors[:, :, None, None]).sum(dim=0)
        rgb = rgb * accum_alpha[None] + self.background * (1.0 - accum_alpha[None])
        depth = (weights * z[:, None, None]).sum(dim=0, keepdim=True)
        return {"rgb": rgb.clamp(0.0, 1.0), "mask": accum_alpha[None], "depth": depth}


def render_shape(rendered: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """Return RGB tensor shape for debug printing."""

    return tuple(rendered["rgb"].shape)
