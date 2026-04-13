"""Train the dynamic Gaussian baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config
from src.pipelines.training import train_model
from src.preprocess.pipeline import run_preprocessing
from src.utils.logging import RunLogger


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_dir", default=None, help="Use an existing run directory with preprocess outputs.")
    parser.add_argument("--input_video", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    overrides = {"preprocess": {}, "train": {}}
    if args.input_video is not None:
        overrides["preprocess"]["input_video"] = args.input_video
    if args.iterations is not None:
        overrides["train"]["iterations"] = args.iterations
    cfg = load_config(args.config, overrides)
    run_dir = Path(args.run_dir) if args.run_dir else cfg.make_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logger = RunLogger(run_dir / "logs.txt")
    if not (run_dir / "preprocess" / "frames").exists():
        run_preprocessing(cfg, run_dir, logger)
    train_model(cfg, run_dir, logger)


if __name__ == "__main__":
    main()
