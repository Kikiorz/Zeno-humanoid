#!/usr/bin/env python3
"""20 Hz fail-closed ROS bridge for Robot8's physical-base V2 policy."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"
DEFAULT_MAPPER = (
    REPO_ROOT
    / "Data"
    / "lerobot"
    / "robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23_base_anchor_odom_v2"
    / "meta"
    / "base_anchor_odom_v2"
    / "dynamics_feedback_mapper.json"
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--cameras"):
        sys.argv.extend(["--cameras", "head_cam,left_arm_cam,right_arm_cam"])
    if not has_option("--rate-hz"):
        sys.argv.extend(["--rate-hz", "20.0"])
    if not has_option("--worker-timeout-s"):
        # A slow synchronous worker must fail to idle rather than leave its
        # previous command active for the legacy one-second socket timeout.
        sys.argv.extend(["--worker-timeout-s", "0.10"])
    if not has_option("--max-obs-age-s"):
        sys.argv.extend(["--max-obs-age-s", "0.10"])
    if not has_option("--base-command-mapper"):
        sys.argv.extend(["--base-command-mapper", str(DEFAULT_MAPPER)])
    if not has_option("--require-base-command-mapper"):
        sys.argv.append("--require-base-command-mapper")
    # These are deliberately independent from the learned V2 physical-action
    # range.  They are conservative low-level limits observed in the source
    # recordings; tune only after dry-run and single-axis robot validation.
    if not has_option("--base-desired-limits"):
        sys.argv.extend(["--base-desired-limits", "0.16,0.16,0.35"])
    if not has_option("--base-command-limits"):
        sys.argv.extend(["--base-command-limits", "0.15,0.15,0.30"])
    if not has_option("--base-command-slew-limits"):
        sys.argv.extend(["--base-command-slew-limits", "0.50,0.50,1.00"])
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
