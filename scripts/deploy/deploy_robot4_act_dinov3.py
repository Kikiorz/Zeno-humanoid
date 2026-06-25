#!/usr/bin/env python3
from __future__ import annotations

from deploy_robot_runtime import parse_worker_args, run_worker


ROBOT = "robot4"
DEFAULT_RUN_ID = "robot4_20260623_act_dinov3_base_dim768"
DEFAULT_WORKER_PORT = 8764


if __name__ == "__main__":
    raise SystemExit(run_worker(ROBOT, parse_worker_args(ROBOT, DEFAULT_RUN_ID, DEFAULT_WORKER_PORT)))
