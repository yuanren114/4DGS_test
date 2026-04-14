"""Run RGB-only preprocessing for a video or image directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config
from src.preprocess.pipeline import run_preprocessing
from src.utils.logging import RunLogger


def _parse_points(value: str | None):
    """Parse `x,y;x,y` point strings."""

    if not value:
        return None
    points = []
    for item in value.split(";"):
        if not item.strip():
            continue
        x, y = item.split(",")
        points.append([float(x), float(y)])
    return points


def _parse_ints(value: str | None):
    """Parse comma-separated integers."""

    if not value:
        return None
    return [int(v) for v in value.split(",") if v.strip()]


def _parse_box(value: str | None):
    """Parse `x0,y0,x1,y1` box strings."""

    if not value:
        return None
    vals = [float(v) for v in value.split(",")]
    if len(vals) != 4:
        raise ValueError("--sam2_box must have four values: x0,y0,x1,y1")
    return vals


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input_video", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--box_threshold", type=float, default=None)
    parser.add_argument(
        "--mask_method",
        default=None,
        choices=["gdino_sam2", "sam2_manual_box", "sam2_manual_mask", "fallback"],
    )
    parser.add_argument("--sam2_config", default=None)
    parser.add_argument("--sam2_ckpt", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop_after_init", action="store_true")
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--sam2_model_cfg", default=None)
    parser.add_argument("--sam2_repo", default=None)
    parser.add_argument("--sam2_box", default=None, help="Manual SAM2 box as 'x0,y0,x1,y1'.")
    parser.add_argument("--sam2_mask_path", default=None, help="Manual first-frame binary mask for sam2_manual_mask.")
    args = parser.parse_args()
    overrides = {"preprocess": {}}
    if args.input_video is not None:
        overrides["preprocess"]["input_video"] = args.input_video
    if args.max_frames is not None:
        overrides["preprocess"]["max_frames"] = args.max_frames
    if args.prompt is not None:
        overrides["preprocess"]["prompt"] = args.prompt
    if args.box_threshold is not None:
        overrides["preprocess"]["box_threshold"] = args.box_threshold
    if args.overwrite:
        overrides["preprocess"]["overwrite"] = True
    if args.stop_after_init:
        overrides["preprocess"]["stop_after_init"] = True
    for key in ["mask_method", "sam2_config", "sam2_ckpt", "sam2_repo"]:
        value = getattr(args, key)
        if value is not None:
            overrides["preprocess"][key] = value
    if args.sam2_checkpoint is not None:
        overrides["preprocess"]["sam2_checkpoint"] = args.sam2_checkpoint
        overrides["preprocess"]["sam2_ckpt"] = args.sam2_checkpoint
    if args.sam2_model_cfg is not None:
        overrides["preprocess"]["sam2_model_cfg"] = args.sam2_model_cfg
        overrides["preprocess"]["sam2_config"] = args.sam2_model_cfg
    if args.sam2_box is not None:
        overrides["preprocess"]["sam2_box"] = _parse_box(args.sam2_box)
    if args.sam2_mask_path is not None:
        overrides["preprocess"]["sam2_mask_path"] = args.sam2_mask_path
    cfg = load_config(args.config, overrides)
    run_dir = cfg.make_run_dir()
    save_config(cfg, run_dir / "config.yaml")
    logger = RunLogger(run_dir / "logs.txt")
    run_preprocessing(cfg, run_dir, logger)


if __name__ == "__main__":
    main()
