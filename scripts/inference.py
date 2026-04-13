"""Run inference from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipelines.inference import run_inference
from src.utils.logging import RunLogger


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "final.pt"
    logger = RunLogger(run_dir / "logs.txt")
    run_inference(cfg, run_dir, checkpoint, logger)


if __name__ == "__main__":
    main()
