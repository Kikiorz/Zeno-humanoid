#!/usr/bin/env python3
"""Deployment entry point for the corrected Robot8 four-camera raw-label ACT model.

The shared worker owns model loading and the calibrated head-stereo path.  This
small wrapper fixes the visual contract for this checkpoint family so it cannot
silently fall back to the legacy unrectified three-camera preprocessing.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_WORKER = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260726_act_dinov3_4cam_640x480_headstereo_rectified_crop_lr_all23_decoder7_b32_100k"
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    if not has_option("--image-size"):
        # The new dataset is 640x480.  The shared worker's calibrated stereo
        # branch letterboxes each 1240x620 eye to this shape without crop.
        sys.argv.extend(["--image-size", "640", "480"])
    if not has_option("--center-crop-fraction"):
        # Arm cameras use direct resize; calibrated head cameras ignore this
        # generic legacy crop entirely.
        sys.argv.extend(["--center-crop-fraction", "1.0"])
    if not has_option("--rectify-head-stereo"):
        sys.argv.append("--rectify-head-stereo")
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
