#!/usr/bin/env python3
"""
One-step deploy: install package + API deps (optional), then start the completeness API.

Usage:
  python scripts/deploy.py
  python scripts/deploy.py --model-dir /path/to/checkpoints
  python scripts/deploy.py --install   # pip install -e ".[api]" then start API

With package installed:
  deploy-completeness   # same, default model-dir=./checkpoints
  deploy-completeness --model-dir /path/to/dir --port 8000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Default: use checkpoints bundled inside the package (so pip install is enough)
try:
    import completeness_nn_api
    _PKG_DIR = Path(completeness_nn_api.__file__).resolve().parent
    DEFAULT_MODEL_DIR = _PKG_DIR / "checkpoints"
except Exception:
    DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deploy cluster-completeness API: optional install, then start server."
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory with the four checkpoint files (default: package checkpoints/)",
    )
    p.add_argument(
        "--outname",
        default="model0",
        help="Model name suffix (default: model0 → best_model_phys_model0.pt, etc.)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API (default: 8000)",
    )
    p.add_argument(
        "--install",
        action="store_true",
        help='Run pip install -e ".[api]" before starting the API',
    )
    args = p.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        print(f"Error: model-dir does not exist: {model_dir}", file=sys.stderr)
        sys.exit(1)

    if args.install:
        print("Installing package with [api] extras...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[api]"],
            cwd=ROOT,
            check=True,
        )

    os.environ["COMPLETENESS_MODEL_DIR"] = str(model_dir)
    os.environ["COMPLETENESS_OUTNAME"] = args.outname
    os.environ["COMPLETENESS_API_HOST"] = "0.0.0.0"
    os.environ["COMPLETENESS_API_PORT"] = str(args.port)

    print(f"Model dir: {model_dir}")
    print(f"Outname:  {args.outname}")
    print(f"API:      http://0.0.0.0:{args.port}/docs")
    print("Starting server...")

    os.chdir(ROOT)
    import uvicorn

    uvicorn.run("completeness_nn_api.serve:app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
