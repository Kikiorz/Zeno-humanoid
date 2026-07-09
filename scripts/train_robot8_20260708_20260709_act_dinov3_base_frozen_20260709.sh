#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV-lerobot-qrp312}"
if ! command -v conda >/dev/null 2>&1 && [[ "${CONDA_ENV}" == "lerobot-qrp312" ]]; then
  CONDA_ENV=""
fi
TRAIN_ENTRY="${REPO_ROOT}/scripts/train_humanmoid_pick_act_dinov2.sh"
RUN_SUFFIX="${RUN_SUFFIX:-20260709}"
RUN_PARALLEL="${RUN_PARALLEL:-true}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-32 24 16 8}"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

COMMON_ENV=(
  USE_AMP=true
  STEPS="${STEPS:-100000}"
  NUM_WORKERS="${NUM_WORKERS:-16}"
  SAVE_FREQ="${SAVE_FREQ:-20000}"
  LOG_FREQ="${LOG_FREQ:-200}"
  VIDEO_BACKEND=pyav
  DINOV2_MODEL=vit_base_patch16_dinov3.lvd1689m
  DINOV2_PRETRAINED=true
  DINOV2_TRAIN_BACKBONE=false
  OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
  OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"
)

run_train() {
  local model_tag="$1"
  local dataset_repo_id="$2"
  local cuda_visible_devices="$3"
  local run_id="robot8_${model_tag}_act_dinov3_base_frozen_100k_640x480_crop2of3_${RUN_SUFFIX}"
  local log_dir="${REPO_ROOT}/outputs/logs"

  mkdir -p "${log_dir}"

  for batch_size in ${BATCH_CANDIDATES}; do
    local log_path="${log_dir}/${run_id}_batch${batch_size}.log"
    echo "[$(date '+%F %T')] start ${run_id} batch=${batch_size} cuda=${cuda_visible_devices}"
    echo "[$(date '+%F %T')] dataset=${dataset_repo_id}"
    echo "[$(date '+%F %T')] log=${log_path}"

    runner=()
    if [[ -n "${CONDA_ENV}" ]]; then
      runner=(conda run --no-capture-output -n "${CONDA_ENV}")
    fi

    if env \
      "${COMMON_ENV[@]}" \
      CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
      DEVICE="cuda" \
      BATCH_SIZE="${batch_size}" \
      RUN_ID="${run_id}" \
      JOB_NAME="${run_id}" \
      DATASET_REPO_ID="${dataset_repo_id}" \
      DATASET_ROOT="${REPO_ROOT}/Data/lerobot/${dataset_repo_id}" \
      OUTPUT_DIR="${REPO_ROOT}/outputs/train/${run_id}" \
      TRAIN_LOG_DIR="${REPO_ROOT}/scripts/train_log/${run_id}" \
      "${runner[@]}" \
      bash "${TRAIN_ENTRY}" \
      2>&1 | tee -a "${log_path}"; then
      echo "[$(date '+%F %T')] done ${run_id} batch=${batch_size}"
      return 0
    fi

    if grep -Eiq "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED" "${log_path}"; then
      echo "[$(date '+%F %T')] OOM at batch=${batch_size}; trying next batch"
      rm -rf "${REPO_ROOT}/outputs/train/${run_id}" "${REPO_ROOT}/scripts/train_log/${run_id}"
      continue
    fi

    echo "[$(date '+%F %T')] non-OOM failure for ${run_id}; see ${log_path}"
    return 1
  done

  echo "[$(date '+%F %T')] all batch candidates failed for ${run_id}"
  return 1
}

if [[ "${RUN_PARALLEL}" == "true" ]]; then
  pids=()
  run_train \
    "20260708_3cam" \
    "robot8_20260708_zeno_h1_auto_cmd_v30_center_crop_2of3_640x480" \
    0 &
  pids+=("$!")

  run_train \
    "20260709_head_right" \
    "robot8_20260709_zeno_h1_auto_cmd_v30_640x480_crop2of3_head_right" \
    1 &
  pids+=("$!")

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  exit "${status}"
fi

run_train \
  "20260708_3cam" \
  "robot8_20260708_zeno_h1_auto_cmd_v30_center_crop_2of3_640x480" \
  0
run_train \
  "20260709_head_right" \
  "robot8_20260709_zeno_h1_auto_cmd_v30_640x480_crop2of3_head_right" \
  1

echo "[$(date '+%F %T')] all robot8 training jobs finished"
