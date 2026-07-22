#!/usr/bin/env python3
"""20 Hz ROS bridge entry point for the ordinary all-23D DINOv3 ACT run."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"
DEFAULT_CAMERAS = "head_cam,left_arm_cam,right_arm_cam"
DEFAULT_RATE_HZ = "20.0"
DEFAULT_WORKER_TIMEOUT_S = "1.0"


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--cameras"):
        sys.argv.extend(["--cameras", DEFAULT_CAMERAS])
    if not has_option("--rate-hz"):
        sys.argv.extend(["--rate-hz", DEFAULT_RATE_HZ])
    if not has_option("--worker-timeout-s"):
        sys.argv.extend(["--worker-timeout-s", DEFAULT_WORKER_TIMEOUT_S])

    # No torso-freezing arguments here: all 23 state/action fields participate
    # in this model's training and deployment.
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
