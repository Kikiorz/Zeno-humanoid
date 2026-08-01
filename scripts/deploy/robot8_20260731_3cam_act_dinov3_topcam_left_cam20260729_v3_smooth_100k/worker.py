#!/usr/bin/env python3
"""Inference entry point for the 2026-07-31 V3 smooth-base ACT+DINOv3 model."""

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
    "base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_bf16_"
    "nogc_resume_100k_onthefly"
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


def reject_fixed_contract_overrides() -> None:
    fixed = (
        "--image-size",
        "--center-crop-fraction",
        "--rectify-head-stereo",
        "--head-stereo-profile",
        "--head-stereo-calibration",
        "--head-stereo-processing",
        "--head-stereo-cam-calibration",
        "--n-action-steps",
    )
    supplied = [name for name in fixed if has_option(name)]
    if supplied:
        raise SystemExit(
            "This V3 deployment has fixed image geometry and requires one new action "
            "per 20 Hz observation; do not override " + ", ".join(supplied)
        )


def main() -> None:
    reject_fixed_contract_overrides()
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    sys.argv.extend(["--image-size", "640", "480"])
    sys.argv.extend(["--center-crop-fraction", "1.0"])
    sys.argv.append("--rectify-head-stereo")
    sys.argv.extend(["--head-stereo-profile", "cam_20260729"])
    sys.argv.extend(["--head-stereo-cam-calibration", str(DEFAULT_CALIBRATION)])
    # V3's base tail is a desired physical odom velocity. Never consume a
    # multi-second stale ACT queue for that feedback-controlled signal.
    sys.argv.extend(["--n-action-steps", "1"])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
