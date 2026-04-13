"""Small logging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import sys


class RunLogger:
    """Mirror important messages to stdout and a run log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """Print and append one message."""

        print(message)
        sys.stdout.flush()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_dict(self, prefix: str, values: Dict[str, float]) -> None:
        """Log a compact dictionary of scalar values."""

        body = ", ".join(f"{k}={v:.6f}" for k, v in values.items())
        self.log(f"{prefix}: {body}")
