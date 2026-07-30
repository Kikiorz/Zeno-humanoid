#!/usr/bin/env bash
# Supervised Vast launcher for the physically anchored Robot8 V2 experiment.
# The cached wrapper keeps the full frozen-DINO memfd descriptor alive until
# this process exits, then tears it down cleanly.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23_base_anchor_odom_v2}"
RUN_ID="${RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base_anchor_odom_v2_base3x_decoder7_ddp128_100k}"

cd "${REPO_ROOT}"
export RUN_ID
export DATASET_REPO_ID
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DINO_CACHE_BATCH_SIZE="${DINO_CACHE_BATCH_SIZE:-32}"
export DINO_CACHE_MIN_FREE_GIB="${DINO_CACHE_MIN_FREE_GIB:-64}"
# Three normalized base dimensions receive 3x loss weight.  Their total loss
# contribution is therefore comparable to the twenty upper-body dimensions.
export ACTION_LOSS_WEIGHTS="${ACTION_LOSS_WEIGHTS:-[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3]}"

exec bash "${REPO_ROOT}/scripts/run_robot8_20260721_act_dinov3_ddp128_all23_cached.sh"
