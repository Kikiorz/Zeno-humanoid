#!/usr/bin/env python3
"""Inference entry point for 2026-07-29 left-topcam ACT checkpoints."""

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
    / "robot8_20260729_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_b32_100k"
)
DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "scripts"
    / "data_convert"
    / "cam"
    / "stereo_params_20260729_172611.npz"
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    if not has_option("--image-size"):
        # Exact training geometry: calibrated left eye -> fixed 1240x620
        # crop -> letterbox to 640x480, with no generic crop or stretch.
        sys.argv.extend(["--image-size", "640", "480"])
    if not has_option("--center-crop-fraction"):
        sys.argv.extend(["--center-crop-fraction", "1.0"])
    if not has_option("--rectify-head-stereo"):
        sys.argv.append("--rectify-head-stereo")
    if not has_option("--head-stereo-profile"):
        sys.argv.extend(["--head-stereo-profile", "cam_20260729"])
    if not has_option("--head-stereo-cam-calibration"):
        sys.argv.extend(["--head-stereo-cam-calibration", str(DEFAULT_CALIBRATION)])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
