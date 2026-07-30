#!/usr/bin/env python3
"""Inference wrapper for the TeaRoom 2026-07-29 ResNet-18 ACT run.

The incoming ROS head image remains the raw 2560x720 left|right fisheye JPEG.
This wrapper pins the camera contract used by the local TeaRoom training run,
then delegates policy loading and inference to the shared ACT worker.
"""

from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_WORKER = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260729_tearoom_act_resnet18_3cam_640x480_topcam_left_cam20260729_all23_b8_10k"
)
DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "scripts"
    / "data_convert"
    / "cam"
    / "stereo_params_20260729_172611.npz"
)
EXPECTED_CALIBRATION_SHA256 = "6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342"

# These settings define the visual model contract, so silently accepting an
# override could make deployment use a different image distribution than the
# trained policy. Checkpoint/device/action scheduling options remain tunable.
PINNED_GEOMETRY_OPTIONS = (
    "--image-size",
    "--center-crop-fraction",
    "--head-stereo-profile",
    "--head-stereo-calibration",
    "--head-stereo-processing",
    "--head-stereo-cam-calibration",
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def verify_calibration() -> None:
    if not DEFAULT_CALIBRATION.is_file():
        raise FileNotFoundError(f"TeaRoom camera calibration is missing: {DEFAULT_CALIBRATION}")
    actual = hashlib.sha256(DEFAULT_CALIBRATION.read_bytes()).hexdigest()
    if actual != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError(
            "TeaRoom camera calibration hash mismatch: "
            f"expected {EXPECTED_CALIBRATION_SHA256}, got {actual}"
        )


def pin_geometry() -> None:
    overridden = [name for name in PINNED_GEOMETRY_OPTIONS if has_option(name)]
    if overridden:
        raise SystemExit(
            "This TeaRoom ResNet wrapper pins the training-time camera geometry; "
            f"do not override: {', '.join(overridden)}"
        )
    if not has_option("--image-size"):
        # Raw stereo -> left-eye rectification -> fixed 1240x620 crop ->
        # letterbox 640x480. No generic centre crop or stretch is permitted.
        sys.argv.extend(["--image-size", "640", "480"])
    if not has_option("--center-crop-fraction"):
        sys.argv.extend(["--center-crop-fraction", "1.0"])
    if not has_option("--rectify-head-stereo"):
        sys.argv.append("--rectify-head-stereo")
    if not has_option("--head-stereo-profile"):
        sys.argv.extend(["--head-stereo-profile", "cam_20260729"])
    if not has_option("--head-stereo-cam-calibration"):
        sys.argv.extend(["--head-stereo-cam-calibration", str(DEFAULT_CALIBRATION)])


def main() -> None:
    verify_calibration()
    pin_geometry()
    if not has_option("--checkpoint-path"):
        # The shared worker resolves the newest complete numbered checkpoint
        # beneath this run directory (currently 005000, then 010000 on finish).
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
