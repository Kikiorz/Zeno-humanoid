#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_WORKER = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260709_head_right_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709"
    / "checkpoints"
    / "020000"
    / "pretrained_model"
)


def main() -> None:
    if not any(arg == "--checkpoint-path" or arg.startswith("--checkpoint-path=") for arg in sys.argv):
        sys.argv.extend(["--checkpoint-path", str(DEFAULT_CHECKPOINT)])
    runpy.run_path(str(BASE_WORKER), run_name="__main__")


if __name__ == "__main__":
    main()
