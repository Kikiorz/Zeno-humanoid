#!/usr/bin/env python3
"""Deployment entry point for the all-23D DINOv3 ACT run with base 3x loss."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_WORKER = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"

# The shared worker selects the highest complete downloaded checkpoint under
# this training-run root.  This run differs from the ordinary counterpart only
# in training loss weights for base_vx, base_vy and base_rotation.
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base3x_decoder7_ddp128_100k"
)
DEFAULT_CENTER_CROP_FRACTION = "1.0"


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    if not has_option("--center-crop-fraction"):
        sys.argv.extend(["--center-crop-fraction", DEFAULT_CENTER_CROP_FRACTION])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")

    # This model is also fully unfrozen.  Do not zero torso state or overwrite
    # its output; only its base loss received extra training weight.
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
