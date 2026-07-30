#!/usr/bin/env python3
"""Inference wrapper for the TeaRoom 2026-07-29 normal ACT+DINOv3 run.

The shared worker owns policy loading and the exact camera preprocessing.  This
wrapper pins only the checkpoint family and the calibrated left-topcam contract
used for TeaRoom training.
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
    / "robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp64_b32_w6_60k"
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
        # The shared worker resolves the latest complete numbered checkpoint
        # below this run directory, e.g. checkpoints/010000/pretrained_model.
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    if not has_option("--image-size"):
        # Training geometry: rectified left RGB -> fixed 1240x620 crop ->
        # 640x480 letterbox.  Do not apply a generic center crop or stretch.
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
