#!/usr/bin/env python3
"""Inference entry point for Data/process-calibrated 2026-07-29 ACT models."""

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
    / "robot8_20260729_act_dinov3_3cam_640x480_topcam_left_dataprocess_v1_all23_decoder7_b32_100k"
)
DATA_PROCESS_CALIBRATION = (
    REPO_ROOT / "configs" / "calibration" / "top_stereo_calibration_basalt_kb4_compat.json"
)
DATA_PROCESS_PROCESSING = (
    REPO_ROOT / "configs" / "calibration" / "processing_metadata_centered_crop_1240x620.json"
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def reject_fixed_geometry_overrides() -> None:
    """Prevent silently running this checkpoint with a different visual contract."""
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
            "This deployment is fixed to the 2026-07-29 Data/process visual "
            "contract; do not override " + ", ".join(supplied)
        )


def main() -> None:
    reject_fixed_geometry_overrides()
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    # Training and deployment share this exact geometry with no generic crop:
    # Data/process rectify/alignment -> 1240x620 crop -> letterbox.
    sys.argv.extend(["--image-size", "640", "480"])
    sys.argv.extend(["--center-crop-fraction", "1.0"])
    sys.argv.append("--rectify-head-stereo")
    sys.argv.extend(["--head-stereo-profile", "data_process_20260729"])
    sys.argv.extend(["--head-stereo-calibration", str(DATA_PROCESS_CALIBRATION)])
    sys.argv.extend(["--head-stereo-processing", str(DATA_PROCESS_PROCESSING)])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
