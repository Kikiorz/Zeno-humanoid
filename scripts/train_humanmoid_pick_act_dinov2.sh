#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

DATASET_REPO_ID="${DATASET_REPO_ID:-humanmoid_pick_zeno_h1_v30}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
JOB_NAME="${JOB_NAME:-humanmoid_pick_act_dinov2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${JOB_NAME}}"

DEVICE="${DEVICE:-cuda}"
USE_AMP="${USE_AMP:-true}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-200}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

DINOV2_MODEL="${DINOV2_MODEL:-vit_small_patch14_dinov2.lvd142m}"
DINOV2_PRETRAINED="${DINOV2_PRETRAINED:-true}"
DINOV2_TRAIN_BACKBONE="${DINOV2_TRAIN_BACKBONE:-false}"

python3 -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --policy.type=act \
  --policy.device="${DEVICE}" \
  --policy.use_amp="${USE_AMP}" \
  --policy.push_to_hub=false \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model="${DINOV2_MODEL}" \
  --policy.dinov2_pretrained="${DINOV2_PRETRAINED}" \
  --policy.dinov2_train_backbone="${DINOV2_TRAIN_BACKBONE}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --wandb.enable=false \
  "$@"
