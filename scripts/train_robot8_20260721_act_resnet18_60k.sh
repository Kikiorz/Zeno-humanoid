#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV-lerobot-qrp312}"

DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260721_zeno_h1_auto_cmd_v30_center_crop_2of3_224x224}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260721_act_resnet18_60k_224x224_crop2of3}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${REPO_ROOT}/scripts/train_log/${RUN_ID}}"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_AMP="${USE_AMP:-true}"
STEPS="${STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-200}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ -z "${ACCELERATE_MIXED_PRECISION:-}" ]]; then
  if [[ "${USE_AMP}" == "true" ]]; then
    export ACCELERATE_MIXED_PRECISION=fp16
  else
    export ACCELERATE_MIXED_PRECISION=no
  fi
fi

runner=()
if [[ -n "${CONDA_ENV}" ]]; then
  runner=(conda run --no-capture-output -n "${CONDA_ENV}")
fi

mkdir -p "${TRAIN_LOG_DIR}"

cat >"${TRAIN_LOG_DIR}/config.env" <<EOF
RUN_ID=${RUN_ID}
DATASET_REPO_ID=${DATASET_REPO_ID}
DATASET_ROOT=${DATASET_ROOT}
OUTPUT_DIR=${OUTPUT_DIR}
DEVICE=${DEVICE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
USE_AMP=${USE_AMP}
ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION}
STEPS=${STEPS}
BATCH_SIZE=${BATCH_SIZE}
NUM_WORKERS=${NUM_WORKERS}
SAVE_FREQ=${SAVE_FREQ}
LOG_FREQ=${LOG_FREQ}
VIDEO_BACKEND=${VIDEO_BACKEND}
VISION_BACKBONE=resnet18
PRETRAINED_BACKBONE_WEIGHTS=ResNet18_Weights.IMAGENET1K_V1
OPTIMIZER_LR=${OPTIMIZER_LR}
OPTIMIZER_LR_BACKBONE=${OPTIMIZER_LR_BACKBONE}
ACTION_LAYOUT=/zeno/h1/auto/wholebody/cmd[1..23]
ACTION_DIM=23
EOF

cat >"${TRAIN_LOG_DIR}/notes.md" <<EOF
# ${RUN_ID}

- Dataset is resampled at 20 Hz from Data/2026_07_21.
- Images use three cameras, center crop 2/3, then resize to 224x224.
- ACT uses an ImageNet-pretrained ResNet-18 vision backbone.
- Dataset action/state layout: /zeno/h1/auto/wholebody/cmd fields [1..23].
- Field [0] control_mode is not trained; deployment should set it to 1.
- Base state uses /zeno/h1/sensor/odom_raw velocity; base action uses /zeno/h1/twist/cmd.
EOF

cmd=(
  "${runner[@]}"
  python3 -m lerobot.scripts.lerobot_train
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --dataset.use_imagenet_stats=true
  --policy.type=act
  --policy.device="${DEVICE}"
  --policy.use_amp="${USE_AMP}"
  --policy.push_to_hub=false
  --policy.vision_backbone=resnet18
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --policy.n_decoder_layers=7
  --policy.optimizer_lr="${OPTIMIZER_LR}"
  --policy.optimizer_lr_backbone="${OPTIMIZER_LR_BACKBONE}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${RUN_ID}"
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
