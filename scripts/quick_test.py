"""Fast end-to-end sanity check for preprocessing, training, and inference."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config
from src.pipelines.inference import run_inference
from src.pipelines.training import train_model
from src.preprocess.pipeline import run_preprocessing
from src.utils.logging import RunLogger


def main() -> None:
    """Run a tiny RGB-only full pipeline."""

    cfg = load_config(
        overrides={
            "quick_test": True,
            "run_name": "run_quick_test",
            "preprocess": {
                "max_frames": 6,
                "image_size": 96,
                "depth_method": "fallback",
                "mask_method": "fallback",
                "track_method": "proxy_grid_lk",
            },
            "model": {"num_gaussians": 96, "num_nodes": 16, "k_nearest_nodes": 4, "init_scale": 0.025},
            "train": {"iterations": 8, "log_every": 1, "save_every": 1000, "device": "auto", "lr": 2.0e-3},
            "inference": {"num_bullet_time_views": 4, "time_index": 2},
        }
    )
    run_dir = cfg.make_run_dir()
    save_config(cfg, run_dir / "config.yaml")
    logger = RunLogger(run_dir / "logs.txt")
    logger.log("quick_test: starting synthetic RGB-only pipeline")
    run_preprocessing(cfg, run_dir, logger)
    ckpt = train_model(cfg, run_dir, logger)
    run_inference(cfg, run_dir, ckpt, logger)
    logger.log(f"quick_test.done: inspect {run_dir}")


if __name__ == "__main__":
    main()
