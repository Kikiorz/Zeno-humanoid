#!/usr/bin/env python3
"""20 Hz ROS bridge for the Robot8 2026-07-26 raw-label ACT baseline."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--cameras"):
        sys.argv.extend(["--cameras", "head_cam,left_arm_cam,right_arm_cam"])
    if not has_option("--rate-hz"):
        sys.argv.extend(["--rate-hz", "20.0"])
    if not has_option("--worker-timeout-s"):
        sys.argv.extend(["--worker-timeout-s", "1.0"])
    # This is the raw-command-label baseline: action[20:23] retains the
    # original low-level base-command semantics, so it deliberately does not
    # enable the V2 physical-velocity feedback mapper.
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
