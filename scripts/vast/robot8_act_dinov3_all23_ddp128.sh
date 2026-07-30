#!/usr/bin/env bash

# Vast supervisor wrapper for the fresh, unfrozen 23-D ACT+DINOv3 run.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
set -euo pipefail
. "${utils}/environment.sh"

exec /workspace/2027icra/scripts/run_robot8_20260721_act_dinov3_ddp128_all23_cached.sh
