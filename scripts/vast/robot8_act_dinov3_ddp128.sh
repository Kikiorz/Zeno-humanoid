#!/usr/bin/env bash

# Vast supervisor wrapper for the remote two-GPU, global-batch-128 ACT+DINOv3 job.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
set -euo pipefail
. "${utils}/environment.sh"

exec /workspace/2027icra/scripts/train_robot8_20260721_act_dinov3_ddp128_resume_60k.sh
