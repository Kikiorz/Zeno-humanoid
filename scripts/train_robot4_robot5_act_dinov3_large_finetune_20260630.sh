#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV-lerobot-qrp312}"
TRAIN_ENTRY="${REPO_ROOT}/scripts/train_humanmoid_pick_act_dinov2.sh"
RUN_SUFFIX="${RUN_SUFFIX:-20260630}"
RUN_PARALLEL="${RUN_PARALLEL:-true}"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# Keep proxy envs if the machine has Clash/V2Ray on 7897; harmless if unused.
export ALL_PROXY="${ALL_PROXY:-socks5://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-socks5://127.0.0.1:7897}"
export HTTP_PROXY="${HTTP_PROXY:-socks5://127.0.0.1:7897}"
export all_proxy="${all_proxy:-${ALL_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"

COMMON_ENV=(
  USE_AMP=true
  STEPS="${STEPS:-100000}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  NUM_WORKERS="${NUM_WORKERS:-12}"
  SAVE_FREQ="${SAVE_FREQ:-20000}"
  LOG_FREQ="${LOG_FREQ:-200}"
  VIDEO_BACKEND=pyav
  DINOV2_MODEL=vit_large_patch16_dinov3.lvd1689m
  DINOV2_PRETRAINED=true
  DINOV2_TRAIN_BACKBONE=true
  OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
  OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-6}"
)

run_train() {
  local robot="$1"
  local dataset_repo_id="$2"
  local cuda_visible_devices="$3"
  shift 3
  local run_id="${robot}_new_act_dinov3_large_finetune_100k_${RUN_SUFFIX}"
  local log_dir="${REPO_ROOT}/outputs/logs"
  local log_path="${log_dir}/${run_id}.log"

  mkdir -p "${log_dir}"

  echo "[$(date '+%F %T')] start ${robot}: ${run_id} cuda_visible_devices=${cuda_visible_devices}"
  echo "[$(date '+%F %T')] dataset: ${dataset_repo_id}"
  echo "[$(date '+%F %T')] log: ${log_path}"

  runner=()
  if [[ -n "${CONDA_ENV}" ]]; then
    runner=(conda run --no-capture-output -n "${CONDA_ENV}")
  fi

  env \
    "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    DEVICE="cuda" \
    RUN_ID="${run_id}" \
    JOB_NAME="${run_id}" \
    DATASET_REPO_ID="${dataset_repo_id}" \
    DATASET_ROOT="${REPO_ROOT}/Data/lerobot/${dataset_repo_id}" \
    OUTPUT_DIR="${REPO_ROOT}/outputs/train/${run_id}" \
    TRAIN_LOG_DIR="${REPO_ROOT}/scripts/train_log/${run_id}" \
    "${runner[@]}" \
    bash "${TRAIN_ENTRY}" \
    "$@" \
    2>&1 | tee -a "${log_path}"

  echo "[$(date '+%F %T')] done ${robot}: ${run_id}"
}

if [[ "${RUN_PARALLEL}" == "true" ]]; then
  pids=()
  run_train robot4 robot4_new_zeno_h1_auto_cmd_v30 0 "$@" &
  pids+=("$!")
  run_train robot5 robot5_new_zeno_h1_auto_cmd_v30 1 "$@" &
  pids+=("$!")

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  exit "${status}"
fi

run_train robot4 robot4_new_zeno_h1_auto_cmd_v30 0 "$@"
run_train robot5 robot5_new_zeno_h1_auto_cmd_v30 1 "$@"

echo "[$(date '+%F %T')] all fine-tuning jobs finished"
