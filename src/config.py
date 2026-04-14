"""Project configuration helpers for the 4DGS360 RGB-only baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as _dt
import json

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    yaml = None


@dataclass
class PreprocessConfig:
    """Settings for RGB video preprocessing."""

    input_video: str = ""
    frame_stride: int = 1
    max_frames: Optional[int] = None
    image_size: int = 256
    depth_method: str = "depth_anything_v2"
    camera_method: str = "colmap_or_simple_vo"
    mask_method: str = "fallback"
    sam2_repo: str = "external/sam2"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_points: List[List[float]] = field(default_factory=list)
    sam2_point_labels: List[int] = field(default_factory=list)
    sam2_box: Optional[List[float]] = None
    selected_mask_id: int = 0
    sam2_obj_id: int = 1
    sam2_mask_threshold: float = 0.0
    track_method: str = "proxy_grid_lk"
    bootstap_repo: str = "external/tapnet"
    bootstap_checkpoint: str = "checkpoints/bootstapir_checkpoint_v2.pt"
    tapip3d_repo: str = "external/TAPIP3D"
    tapip3d_checkpoint: str = "checkpoints/tapip3d_final.pth"
    tapip3d_resolution_factor: int = 1
    tapip3d_num_iters: int = 6
    tapip3d_support_grid_size: int = 16
    reuse_existing: bool = True


@dataclass
class ModelConfig:
    """Settings for the dynamic Gaussian model."""

    num_gaussians: int = 512
    num_nodes: int = 64
    k_nearest_nodes: int = 4
    sh_degree: int = 0
    init_scale: float = 0.035
    background: float = 1.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    iterations: int = 1000
    batch_size: int = 1
    lr: float = 1.0e-3
    rgb_weight: float = 0.8
    dssim_weight: float = 0.2
    lpips_weight: float = 0.0
    mask_weight: float = 1.0
    depth_weight: float = 0.5
    depth_smooth_weight: float = 0.05
    track_weight: float = 0.2
    arap_weight: float = 0.1
    log_every: int = 10
    save_every: int = 250
    device: str = "auto"


@dataclass
class InferenceConfig:
    """Inference and visualization options."""

    checkpoint_path: str = ""
    render_size: int = 256
    num_bullet_time_views: int = 24
    time_index: int = 0


@dataclass
class AppConfig:
    """Top-level project configuration."""

    output_root: str = "outputs"
    run_name: str = ""
    quick_test: bool = False
    seed: int = 7
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def make_run_dir(self) -> Path:
        """Create and return a timestamped run directory."""

        if not self.run_name:
            self.run_name = "run_" + _dt.datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = Path(self.output_root) / self.run_name
        for name in ["preprocess", "checkpoints", "debug", "visualizations", "inference"]:
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        return run_dir


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    """Load configuration from YAML plus optional dictionary overrides."""

    cfg_dict = asdict(AppConfig())
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) if yaml is not None else json.load(f)
            loaded = loaded or {}
        cfg_dict = _merge_dict(cfg_dict, loaded)
    if overrides:
        cfg_dict = _merge_dict(cfg_dict, overrides)

    return AppConfig(
        output_root=cfg_dict["output_root"],
        run_name=cfg_dict["run_name"],
        quick_test=cfg_dict["quick_test"],
        seed=cfg_dict["seed"],
        preprocess=PreprocessConfig(**cfg_dict["preprocess"]),
        model=ModelConfig(**cfg_dict["model"]),
        train=TrainConfig(**cfg_dict["train"]),
        inference=InferenceConfig(**cfg_dict["inference"]),
    )


def save_config(config: AppConfig, path: Path) -> None:
    """Write a configuration object to YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(asdict(config), f, sort_keys=False)
        else:
            json.dump(asdict(config), f, indent=2)
