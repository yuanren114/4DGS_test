"""Clone official tracker repositories and optionally download checkpoints."""

from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import urllib.request


REPOS = {
    "tapnet": "https://github.com/google-deepmind/tapnet.git",
    "TAPIP3D": "https://github.com/zbw001/TAPIP3D.git",
}

CHECKPOINTS = {
    "checkpoints/bootstapir_checkpoint_v2.pt": "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",
    "checkpoints/tapip3d_final.pth": "https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth",
}


def clone_repo(name: str, url: str, external_dir: Path) -> None:
    """Clone one external repository if it is missing."""

    target = external_dir / name
    if target.exists():
        print(f"exists: {target}")
        return
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)


def download_file(path: Path, url: str) -> None:
    """Download one checkpoint if it is missing."""

    if path.exists():
        print(f"exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    external_dir = root / "external"
    external_dir.mkdir(exist_ok=True)
    for name, url in REPOS.items():
        clone_repo(name, url, external_dir)
    if not args.skip_checkpoints:
        for rel_path, url in CHECKPOINTS.items():
            download_file(root / rel_path, url)
    print("External tracker setup finished.")


if __name__ == "__main__":
    sys.exit(main())
