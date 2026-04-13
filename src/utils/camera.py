"""Camera math used by preprocessing, rendering, and losses."""

from __future__ import annotations

from typing import Tuple

import torch


def default_intrinsics(width: int, height: int, fov_degrees: float = 60.0) -> torch.Tensor:
    """Create a pinhole intrinsic matrix from image size and horizontal field of view."""

    f = 0.5 * width / torch.tan(torch.tensor(0.5 * fov_degrees * 3.14159265 / 180.0))
    k = torch.eye(3, dtype=torch.float32)
    k[0, 0] = f
    k[1, 1] = f
    k[0, 2] = (width - 1) * 0.5
    k[1, 2] = (height - 1) * 0.5
    return k


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    """Return a world-to-camera matrix for a camera looking at target."""

    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    z = target - eye
    z = z / (torch.linalg.norm(z) + 1.0e-8)
    x = torch.cross(z, up, dim=0)
    x = x / (torch.linalg.norm(x) + 1.0e-8)
    y = torch.cross(x, z, dim=0)
    r = torch.stack([x, y, z], dim=0)
    t = -r @ eye
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = r
    pose[:3, 3] = t
    return pose


def project_points(points_world: torch.Tensor, intrinsics: torch.Tensor, world_to_cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points to pixel coordinates and return depth."""

    ones = torch.ones((points_world.shape[0], 1), dtype=points_world.dtype, device=points_world.device)
    homo = torch.cat([points_world, ones], dim=-1)
    cam = (world_to_cam.to(points_world.device, points_world.dtype) @ homo.T).T[:, :3]
    z = cam[:, 2].clamp_min(1.0e-4)
    uv = (intrinsics.to(points_world.device, points_world.dtype) @ cam.T).T
    uv = uv[:, :2] / z[:, None]
    return uv, z


def unproject_pixels(uv: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor, cam_to_world: torch.Tensor) -> torch.Tensor:
    """Unproject pixels with depth into world coordinates."""

    k_inv = torch.inverse(intrinsics.to(uv.device, uv.dtype))
    ones = torch.ones((uv.shape[0], 1), dtype=uv.dtype, device=uv.device)
    pix = torch.cat([uv, ones], dim=-1)
    cam = (k_inv @ pix.T).T * depth[:, None]
    homo = torch.cat([cam, ones], dim=-1)
    world = (cam_to_world.to(uv.device, uv.dtype) @ homo.T).T[:, :3]
    return world
