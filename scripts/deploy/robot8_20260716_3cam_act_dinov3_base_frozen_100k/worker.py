#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_WORKER = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"
DEFAULT_CENTER_CROP_FRACTION = "0.6666667"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260716_act_dinov3_base_frozen_100k_640x480_crop2of3_curated30"
    / "checkpoints"
    / "100000"
    / "pretrained_model"
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT)])
    if not has_option("--center-crop-fraction"):
        sys.argv.extend(["--center-crop-fraction", DEFAULT_CENTER_CROP_FRACTION])
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
