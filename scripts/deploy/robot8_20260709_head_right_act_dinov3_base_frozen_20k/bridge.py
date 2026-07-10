#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BRIDGE = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "bridge.py"
DEFAULT_CAMERAS = "head_cam,right_arm_cam"


def main() -> None:
    if not any(arg == "--cameras" or arg.startswith("--cameras=") for arg in sys.argv):
        sys.argv.extend(["--cameras", DEFAULT_CAMERAS])
    runpy.run_path(str(BASE_BRIDGE), run_name="__main__")


if __name__ == "__main__":
    main()
