#!/usr/bin/env python3
"""20 Hz ROS bridge for the 2026-07-31 raw-command ACT+DINOv3 model."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"
THREE_CAMERAS = "head_cam,left_arm_cam,right_arm_cam"


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def reject_fixed_stream_overrides() -> None:
    fixed = ("--cameras", "--rate-hz", "--head-cam-topic")
    supplied = [name for name in fixed if has_option(name)]
    if supplied:
        raise SystemExit(
            "This deployment requires the raw 2560x720 head stereo topic, 20 Hz, "
            "and head_cam,left_arm_cam,right_arm_cam; do not override " + ", ".join(supplied)
        )


def main() -> None:
    reject_fixed_stream_overrides()
    # head_cam is the unmodified left|right 2560x720 JPEG. The worker alone
    # constructs the calibrated left-eye model input.
    sys.argv.extend(["--cameras", THREE_CAMERAS])
    sys.argv.extend(["--rate-hz", "20.0"])
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
