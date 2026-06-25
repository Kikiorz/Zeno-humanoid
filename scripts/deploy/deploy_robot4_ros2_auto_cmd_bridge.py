#!/usr/bin/env python3
from __future__ import annotations

from deploy_robot_runtime import parse_bridge_args, run_bridge


ROBOT = "robot4"
DEFAULT_WORKER_PORT = 8764


if __name__ == "__main__":
    raise SystemExit(run_bridge(ROBOT, parse_bridge_args(ROBOT, DEFAULT_WORKER_PORT)))
