#!/usr/bin/env python3
"""20 Hz fail-closed ROS bridge for the 2026-07-31 V3 smooth-base model."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"
THREE_CAMERAS = "head_cam,left_arm_cam,right_arm_cam"
DEFAULT_MAPPER = Path(__file__).resolve().with_name("dynamics_feedback_mapper.json")


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def reject_fixed_contract_overrides() -> None:
    fixed = (
        "--cameras",
        "--rate-hz",
        "--head-cam-topic",
        "--worker-timeout-s",
        "--max-obs-age-s",
        "--base-command-mapper",
        "--require-base-command-mapper",
        "--base-desired-limits",
        "--base-command-limits",
        "--base-command-slew-limits",
    )
    supplied = [name for name in fixed if has_option(name)]
    if supplied:
        raise SystemExit(
            "This V3 deployment has a fixed raw-camera/20-Hz/feedback-mapper "
            "contract; do not override " + ", ".join(supplied)
        )


def main() -> None:
    reject_fixed_contract_overrides()
    if not DEFAULT_MAPPER.is_file():
        raise SystemExit(f"Missing required V3 dynamics mapper: {DEFAULT_MAPPER}")
    # head_cam remains the raw 2560x720 stereo JPEG; the worker materializes
    # only the calibrated left RGB model input.
    sys.argv.extend(["--cameras", THREE_CAMERAS])
    sys.argv.extend(["--rate-hz", "20.0"])
    # A stale observation or an inference cycle slower than the physical
    # feedback period fails closed and publishes idle rather than an old twist.
    sys.argv.extend(["--worker-timeout-s", "0.10"])
    sys.argv.extend(["--max-obs-age-s", "0.10"])
    sys.argv.extend(["--base-command-mapper", str(DEFAULT_MAPPER)])
    sys.argv.append("--require-base-command-mapper")
    sys.argv.extend(["--base-desired-limits", "0.16,0.16,0.35"])
    sys.argv.extend(["--base-command-limits", "0.15,0.15,0.30"])
    sys.argv.extend(["--base-command-slew-limits", "0.50,0.50,1.00"])
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
