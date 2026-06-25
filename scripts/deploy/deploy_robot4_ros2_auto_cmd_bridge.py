#!/usr/bin/env python3
from __future__ import annotations

import sys

from human_new_pick_ros2_auto_cmd_bridge import main


DEFAULT_ROS_ARGS = [
    "--ros-args",
    "-r",
    "__node:=robot4_auto_cmd_bridge",
    "-p",
    "worker_port:=8764",
    "-p",
    "publish_commands:=false",
]


if __name__ == "__main__":
    if "--ros-args" not in sys.argv:
        sys.argv.extend(DEFAULT_ROS_ARGS)
    main()
