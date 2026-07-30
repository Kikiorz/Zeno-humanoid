#!/usr/bin/env bash

# Vast supervisor wrapper for the strictly matched base-weighted counterpart
# of the all-23D ACT+DINOv3 run.  The generic cached runner supplies every
# other training parameter; only the final three action dimensions differ.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
set -euo pipefail
. "${utils}/environment.sh"

export RUN_ID="${RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base3x_decoder7_ddp128_100k}"
# action indices 20--22: base_vx, base_vy, base_rotation.
export ACTION_LOSS_WEIGHTS="${ACTION_LOSS_WEIGHTS:-[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3]}"

exec /workspace/2027icra/scripts/run_robot8_20260721_act_dinov3_ddp128_all23_cached.sh
