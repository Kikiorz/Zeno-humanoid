#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_WORKER = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"
DEFAULT_CENTER_CROP_FRACTION = "1.0"
DEFAULT_FROZEN_FIELDS = "torso_lift,torso_waist"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260721_act_dinov3_3cam_640x480_nocrop_frozen_lift_waist_base3x_decoder7_10k"
    / "checkpoints"
    / "010000"
    / "pretrained_model"
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT)])
    if not has_option("--center-crop-fraction"):
        sys.argv.extend(["--center-crop-fraction", DEFAULT_CENTER_CROP_FRACTION])
    if not has_option("--frozen-fields"):
        sys.argv.extend(["--frozen-fields", DEFAULT_FROZEN_FIELDS])
    if not has_option("--use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
