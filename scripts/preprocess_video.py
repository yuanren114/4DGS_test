"""Run RGB-only preprocessing for a video or image directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config
from src.preprocess.pipeline import run_preprocessing
from src.utils.logging import RunLogger


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input_video", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()
    overrides = {"preprocess": {}}
    if args.input_video is not None:
        overrides["preprocess"]["input_video"] = args.input_video
    if args.max_frames is not None:
        overrides["preprocess"]["max_frames"] = args.max_frames
    cfg = load_config(args.config, overrides)
    run_dir = cfg.make_run_dir()
    save_config(cfg, run_dir / "config.yaml")
    logger = RunLogger(run_dir / "logs.txt")
    run_preprocessing(cfg, run_dir, logger)


if __name__ == "__main__":
    main()
