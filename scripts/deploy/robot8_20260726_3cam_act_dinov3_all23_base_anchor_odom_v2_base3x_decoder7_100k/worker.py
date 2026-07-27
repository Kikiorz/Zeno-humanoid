#!/usr/bin/env python3
"""Inference entry point for the Robot8 2026-07-26 V2 base-weighted ACT run."""

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
    / "robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_base_anchor_odom_v2_base3x_decoder7_b32x2_100k"
)


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    if not has_option("--center-crop-fraction"):
        sys.argv.extend(["--center-crop-fraction", "1.0"])
    if not has_option("--n-action-steps"):
        # V2's base tail is a physical desired velocity.  Re-evaluate it every
        # control tick rather than consuming a five-second stale ACT queue.
        sys.argv.extend(["--n-action-steps", "1"])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
