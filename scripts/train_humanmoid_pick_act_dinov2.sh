#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

DATASET_REPO_ID="${DATASET_REPO_ID:-humanmoid_pick_zeno_h1_auto_cmd_v30}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
JOB_NAME="${JOB_NAME:-humanmoid_pick_act_dinov2_auto_cmd23}"
RUN_ID="${RUN_ID:-${JOB_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${REPO_ROOT}/scripts/train_log}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${TRAIN_LOG_ROOT}/${RUN_ID}}"

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
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"

mkdir -p "${TRAIN_LOG_DIR}"

cat >"${TRAIN_LOG_DIR}/config.env" <<EOF
RUN_ID=${RUN_ID}
DATASET_REPO_ID=${DATASET_REPO_ID}
DATASET_ROOT=${DATASET_ROOT}
OUTPUT_DIR=${OUTPUT_DIR}
JOB_NAME=${JOB_NAME}
DEVICE=${DEVICE}
USE_AMP=${USE_AMP}
STEPS=${STEPS}
BATCH_SIZE=${BATCH_SIZE}
NUM_WORKERS=${NUM_WORKERS}
SAVE_FREQ=${SAVE_FREQ}
LOG_FREQ=${LOG_FREQ}
VIDEO_BACKEND=${VIDEO_BACKEND}
DINOV2_MODEL=${DINOV2_MODEL}
DINOV2_PRETRAINED=${DINOV2_PRETRAINED}
DINOV2_TRAIN_BACKBONE=${DINOV2_TRAIN_BACKBONE}
OPTIMIZER_LR=${OPTIMIZER_LR}
OPTIMIZER_LR_BACKBONE=${OPTIMIZER_LR_BACKBONE}
ACTION_LAYOUT=/zeno/h1/auto/wholebody/cmd[1..23]
ACTION_DIM=23
EOF

cat >"${TRAIN_LOG_DIR}/notes.md" <<EOF
# ${RUN_ID}

- Dataset action/state layout: /zeno/h1/auto/wholebody/cmd fields [1..23].
- Field [0] control_mode is not trained; deployment should set it to 1.
- Base state uses /zeno/h1/sensor/odom_raw velocity; base action uses /zeno/h1/twist/cmd.
- Model checkpoints are written under: ${OUTPUT_DIR}
EOF

cmd=(
  python3 -m lerobot.scripts.lerobot_train
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --policy.type=act
  --policy.device="${DEVICE}"
  --policy.use_amp="${USE_AMP}"
  --policy.push_to_hub=false
  --policy.vision_backbone=dinov2
  --policy.dinov2_model="${DINOV2_MODEL}"
  --policy.dinov2_pretrained="${DINOV2_PRETRAINED}"
  --policy.dinov2_train_backbone="${DINOV2_TRAIN_BACKBONE}"
  --policy.optimizer_lr="${OPTIMIZER_LR}"
  --policy.optimizer_lr_backbone="${OPTIMIZER_LR_BACKBONE}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --steps="${STEPS}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --wandb.enable=false
)

printf '%q ' "${cmd[@]}" "$@" >"${TRAIN_LOG_DIR}/command.txt"
printf '\n' >>"${TRAIN_LOG_DIR}/command.txt"

"${cmd[@]}" "$@" 2>&1 | tee -a "${TRAIN_LOG_DIR}/train.log"
