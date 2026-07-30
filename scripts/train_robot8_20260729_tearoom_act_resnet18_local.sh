#!/usr/bin/env bash
# Local single-GPU ACT baseline for the rectified TeaRoom three-camera dataset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot-qrp312}"

DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260729_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260729_tearoom_act_resnet18_3cam_640x480_topcam_left_cam20260729_all23_b8_10k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${REPO_ROOT}/outputs/logs/${RUN_ID}}"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_AMP="${USE_AMP:-true}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  printf 'Missing converted TeaRoom dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite an existing training output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ && "${SAVE_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'STEPS and SAVE_FREQ must be positive integers.\n' >&2
  exit 1
fi
if (( STEPS % SAVE_FREQ != 0 )); then
  printf 'SAVE_FREQ must divide STEPS for exact checkpoint boundaries.\n' >&2
  exit 1
fi
if [[ "${BATCH_SIZE}" != "8" ]]; then
  printf 'This local preset is intentionally batch size 8; got BATCH_SIZE=%s.\n' "${BATCH_SIZE}" >&2
  exit 1
fi

runner=(python3)
if [[ -n "${CONDA_ENV}" ]]; then
  runner=(conda run --no-capture-output -n "${CONDA_ENV}" python3)
fi

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
mkdir -p "${TRAIN_LOG_DIR}"

"${runner[@]}" - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
if info.get("total_episodes") != 18 or info.get("total_frames") != 37546:
    raise SystemExit("TeaRoom dataset episode/frame contract is not satisfied")
topcam = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
if topcam.get("profile") != "cam_20260729" or topcam.get("selected_model_topcam_eye") != "left":
    raise SystemExit("TeaRoom dataset is not the requested cam_20260729 rectified source")
print("Verified rectified TeaRoom source dataset for local ResNet-18 training.", flush=True)
PY

cmd=(
  "${runner[@]}"
  -m lerobot.scripts.lerobot_train
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
  --prefetch_factor="${PREFETCH_FACTOR}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --eval_freq=0
  --wandb.enable=false
)

printf '[%s] local ACT ResNet-18: run=%s steps=%s batch=%s save_freq=%s\n' \
  "$(date '+%F %T')" "${RUN_ID}" "${STEPS}" "${BATCH_SIZE}" "${SAVE_FREQ}" | tee -a "${TRAIN_LOG_DIR}/train.log"
printf '%q ' "${cmd[@]}" >"${TRAIN_LOG_DIR}/command.txt"
printf '\n' >>"${TRAIN_LOG_DIR}/command.txt"
"${cmd[@]}" "$@" 2>&1 | tee -a "${TRAIN_LOG_DIR}/train.log"
touch "${OUTPUT_DIR}/TRAINING_SUCCEEDED"
