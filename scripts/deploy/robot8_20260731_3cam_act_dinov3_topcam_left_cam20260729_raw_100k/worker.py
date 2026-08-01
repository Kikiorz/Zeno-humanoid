#!/usr/bin/env python3
"""Inference entry point for the 2026-07-31 raw-command ACT+DINOv3 model."""

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
    / "robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_"
    "decoder7_b32_bf16_nogc_resume_100k_onthefly"
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


def reject_fixed_geometry_overrides() -> None:
    """Keep deployment pixels exactly equal to the training pixel contract."""

    fixed = (
        "--image-size",
        "--center-crop-fraction",
        "--rectify-head-stereo",
        "--head-stereo-profile",
        "--head-stereo-calibration",
        "--head-stereo-processing",
        "--head-stereo-cam-calibration",
    )
    supplied = [name for name in fixed if has_option(name)]
    if supplied:
        raise SystemExit(
            "This model has a fixed cam_20260729 image contract; do not override "
            + ", ".join(supplied)
        )


def main() -> None:
    reject_fixed_geometry_overrides()
    if not has_option("--checkpoint-path"):
        # Passing a run root makes the shared worker pick its latest numbered
        # checkpoint. Pass --checkpoint-path explicitly to test 040000, etc.
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    # raw 2560x720 stereo -> rectify/align -> crop -> left RGB -> letterbox.
    sys.argv.extend(["--image-size", "640", "480"])
    sys.argv.extend(["--center-crop-fraction", "1.0"])
    sys.argv.append("--rectify-head-stereo")
    sys.argv.extend(["--head-stereo-profile", "cam_20260729"])
    sys.argv.extend(["--head-stereo-cam-calibration", str(DEFAULT_CALIBRATION)])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
