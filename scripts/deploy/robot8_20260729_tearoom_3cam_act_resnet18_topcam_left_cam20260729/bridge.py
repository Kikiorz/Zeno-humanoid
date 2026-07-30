#!/usr/bin/env python3
"""20 Hz ROS bridge wrapper for the TeaRoom ResNet-18 ACT policy."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"
THREE_CAMERAS = "head_cam,left_arm_cam,right_arm_cam"


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    # Keep head_cam as the raw 2560x720 side-by-side JPEG topic.  The worker,
    # not this ROS bridge, applies the exact calibrated left-eye preprocessing.
    if not has_option("--cameras"):
        sys.argv.extend(["--cameras", THREE_CAMERAS])
    if not has_option("--rate-hz"):
        sys.argv.extend(["--rate-hz", "20.0"])
    if not has_option("--worker-timeout-s"):
        sys.argv.extend(["--worker-timeout-s", "1.0"])
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
