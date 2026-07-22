#!/usr/bin/env python3
"""Deployment entry point for the unfrozen, ordinary all-23D DINOv3 ACT run."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_WORKER = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"

# Pass the training-run directory rather than a numbered snapshot.  The shared
# worker resolves the highest *complete* ``checkpoints/*/pretrained_model``
# directory, so a newer downloaded checkpoint becomes the default without
# editing deployment code.
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_decoder7_ddp128_100k"
)
DEFAULT_CENTER_CROP_FRACTION = "1.0"


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not has_option("--checkpoint-path"):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT_ROOT)])
    if not has_option("--center-crop-fraction"):
        sys.argv.extend(["--center-crop-fraction", DEFAULT_CENTER_CROP_FRACTION])
    if not has_option("--use-amp") and not has_option("--no-use-amp"):
        sys.argv.append("--use-amp")

    # This is the all-23D model: do not pass --frozen-fields or fixed action
    # values.  Real torso state is an input and all 23 action values are model
    # outputs, exactly as in this training run.
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
