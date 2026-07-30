#!/usr/bin/env python3
"""20 Hz ROS bridge entry point for the corrected Robot8 four-camera raw model."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"
FOUR_CAMERAS = "head_cam,head_cam_right,left_arm_cam,right_arm_cam"


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--cameras"):
        # `head_cam` and `head_cam_right` deliberately refer to the same raw
        # compressed stereo topic.  The worker splits that one JPEG into the
        # calibrated independent left/right model inputs.
        sys.argv.extend(["--cameras", FOUR_CAMERAS])
    if not has_option("--rate-hz"):
        sys.argv.extend(["--rate-hz", "20.0"])
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
