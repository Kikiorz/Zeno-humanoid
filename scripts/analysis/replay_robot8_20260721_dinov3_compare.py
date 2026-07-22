#!/usr/bin/env python3
"""Run the data03 ACT base-command comparison for the current DINOv3 run.

This is intentionally a thin configuration wrapper around
``replay_robot8_act_base_compare.py``.  The shared replay implementation
already reproduces bridge-style causal input caching at 20 Hz, writes one PNG
and CSV per checkpoint, and integrates the last three action dimensions as a
holonomic base command.  This wrapper supplies the current model's deployment
defaults without changing the older analysis file.

RTC terminology for this ACT policy:
  * ``RTC off`` is the deployed default: 100-action FIFO chunks at 20 Hz.
  * ``RTC on`` is ACT's usable every-tick alternative: temporal ensemble with
    ``n_action_steps=1`` and coefficient 0.01.  Generic LeRobot RTC is not
    supported by ACT because it is only implemented for flow-matching models.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "scripts" / "analysis" / "replay_robot8_act_base_compare.py"
RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260721_act_dinov3_3cam_640x480_nocrop_frozen_lift_waist_decoder7_10k"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "robot8_20260721_data03_act_dinov3_decoder7_compare"
FROZEN_ACTION_INDICES = (0, 1)  # torso_lift, torso_waist
FROZEN_ACTION_VALUES = {
    0: -0.001354230436173755,
    1: -0.06551777579140391,
}


def load_shared_module() -> Any:
    spec = importlib.util.spec_from_file_location("robot8_shared_act_replay", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import shared replay implementation: {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    replay = load_shared_module()
    replay.RUN_DIR = RUN_DIR
    replay.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    def make_worker(checkpoint: Path, *, temporal_ensemble: bool, device: str | None):
        # This directly instantiates the same ActWorker used by deployment.
        # The dedicated deployment wrapper sets these exact values by default:
        # 640x480 checkpoint input, no crop, CUDA AMP, clamps, and frozen torso
        # lift/waist state and action fields.
        return replay.DEPLOY_WORKER.ActWorker(
            checkpoint=checkpoint,
            device=device,
            image_size=None,
            center_crop_fraction=1.0,
            use_amp=True,
            clamp_actions=True,
            action_clip_margin=0.05,
            n_action_steps=1 if temporal_ensemble else None,
            temporal_ensemble_coeff=0.01 if temporal_ensemble else None,
            frozen_action_indices=FROZEN_ACTION_INDICES,
            fixed_action_values=FROZEN_ACTION_VALUES,
        )

    replay.make_worker = make_worker
    replay.main()


if __name__ == "__main__":
    main()
