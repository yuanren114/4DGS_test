"""Dynamic Gaussian model with node-based motion."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.renderer import GaussianRenderer


class DynamicGaussianModel(nn.Module):
    """A compact PyTorch approximation of 4DGS360's dynamic Gaussian representation.

    The implementation keeps the paper's canonical Gaussians plus node-guided
    temporal deformation structure, but uses a lightweight renderer and
    per-frame node translations so it remains runnable in a small repository.
    """

    def __init__(self, config: ModelConfig, num_frames: int, height: int, width: int) -> None:
        super().__init__()
        self.config = config
        self.num_frames = num_frames
        self.renderer = GaussianRenderer(height, width, background=config.background)
        self.means = nn.Parameter(torch.randn(config.num_gaussians, 3) * 0.25 + torch.tensor([0.0, 0.0, 1.3]))
        self.log_scales = nn.Parameter(torch.full((config.num_gaussians, 1), torch.log(torch.tensor(config.init_scale))))
        self.color_logits = nn.Parameter(torch.randn(config.num_gaussians, 3) * 0.1)
        self.opacity_logits = nn.Parameter(torch.full((config.num_gaussians, 1), 0.0))
        self.node_positions = nn.Parameter(torch.randn(config.num_nodes, 3) * 0.3 + torch.tensor([0.0, 0.0, 1.3]))
        self.node_offsets = nn.Parameter(torch.zeros(num_frames, config.num_nodes, 3))

    def initialize_from_tracks(self, tracks_xyz: torch.Tensor, images: torch.Tensor) -> None:
        """Initialize Gaussian means and colors from pseudo-3D tracks."""

        with torch.no_grad():
            xyz0 = tracks_xyz[0]
            n = min(self.means.shape[0], xyz0.shape[0])
            self.means[:n].copy_(xyz0[:n])
            if n < self.means.shape[0]:
                repeat = xyz0[torch.randint(0, xyz0.shape[0], (self.means.shape[0] - n,))]
                self.means[n:].copy_(repeat + 0.01 * torch.randn_like(repeat))
            self.node_positions.copy_(self.means[torch.linspace(0, self.means.shape[0] - 1, self.config.num_nodes).long()])
            rgb = images[0].permute(1, 2, 0).reshape(-1, 3)
            colors = rgb[torch.linspace(0, rgb.shape[0] - 1, self.means.shape[0]).long()]
            self.color_logits.copy_(torch.logit(colors.clamp(0.01, 0.99)))

    def deformed_means(self, time_index: torch.Tensor | int) -> torch.Tensor:
        """Apply K-nearest node translations to canonical means."""

        t = int(time_index.item()) if torch.is_tensor(time_index) else int(time_index)
        distances = torch.cdist(self.means, self.node_positions).clamp_min(1.0e-6)
        k = min(self.config.k_nearest_nodes, self.config.num_nodes)
        vals, inds = torch.topk(distances, k=k, largest=False)
        weights = torch.softmax(-vals / (vals.mean().detach() + 1.0e-6), dim=-1)
        offsets = self.node_offsets[t, inds]
        motion = (weights[..., None] * offsets).sum(dim=1)
        return self.means + motion

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Render one sample."""

        t = sample["time_index"]
        means_t = self.deformed_means(t)
        colors = torch.sigmoid(self.color_logits)
        opacity = torch.sigmoid(self.opacity_logits).squeeze(-1)
        scales = torch.exp(self.log_scales)
        rendered = self.renderer(means_t, colors, opacity, scales, sample["intrinsics"], sample["world_to_camera"])
        rendered["means_t"] = means_t
        return rendered
