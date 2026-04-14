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
    parser.add_argument("--mask_method", default=None, choices=["fallback", "sam2_manual_init", "sam2_auto_first_frame"])
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--sam2_model_cfg", default=None)
    parser.add_argument("--sam2_repo", default=None)
    parser.add_argument("--sam2_points", default=None, help="Manual SAM2 points as 'x,y;x,y'.")
    parser.add_argument("--sam2_point_labels", default=None, help="Manual SAM2 labels as '1,0,...'.")
    parser.add_argument("--sam2_box", default=None, help="Manual SAM2 box as 'x0,y0,x1,y1'.")
    parser.add_argument("--selected_mask_id", type=int, default=None)
    args = parser.parse_args()
    overrides = {"preprocess": {}}
    if args.input_video is not None:
        overrides["preprocess"]["input_video"] = args.input_video
    if args.max_frames is not None:
        overrides["preprocess"]["max_frames"] = args.max_frames
    for key in ["mask_method", "sam2_checkpoint", "sam2_model_cfg", "sam2_repo"]:
        value = getattr(args, key)
        if value is not None:
            overrides["preprocess"][key] = value
    if args.sam2_points is not None:
        overrides["preprocess"]["sam2_points"] = _parse_points(args.sam2_points)
    if args.sam2_point_labels is not None:
        overrides["preprocess"]["sam2_point_labels"] = _parse_ints(args.sam2_point_labels)
    if args.sam2_box is not None:
        overrides["preprocess"]["sam2_box"] = _parse_box(args.sam2_box)
    if args.selected_mask_id is not None:
        overrides["preprocess"]["selected_mask_id"] = args.selected_mask_id
    cfg = load_config(args.config, overrides)
    run_dir = cfg.make_run_dir()
    save_config(cfg, run_dir / "config.yaml")
    logger = RunLogger(run_dir / "logs.txt")
    run_preprocessing(cfg, run_dir, logger)


if __name__ == "__main__":
    main()
