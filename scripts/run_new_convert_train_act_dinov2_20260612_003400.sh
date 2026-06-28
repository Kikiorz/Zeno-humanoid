#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/zeno-rp/2027icra"
RUN_ID="new_act_dinov2_dinoft_20260612_003400"
DATASET_REPO_ID="new_zeno_h1_v30"
DATA_DIR="${REPO_ROOT}/Data/new"
DATASET_ROOT="${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}"
OUTPUT_DIR="${REPO_ROOT}/outputs/train/${RUN_ID}"
LOG_DIR="${REPO_ROOT}/outputs/logs"
PIPELINE_LOG="${LOG_DIR}/${RUN_ID}_pipeline.log"
CONDA_BIN="/home/zeno-rp/miniconda3/bin/conda"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/Data/lerobot"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export ALL_PROXY="${ALL_PROXY:-socks5://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-socks5://127.0.0.1:7897}"
export HTTP_PROXY="${HTTP_PROXY:-socks5://127.0.0.1:7897}"
export all_proxy="${all_proxy:-${ALL_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"

echo "[pipeline] start $(date --iso-8601=seconds)"
echo "[pipeline] data_dir=${DATA_DIR}"
echo "[pipeline] dataset_root=${DATASET_ROOT}"
echo "[pipeline] output_dir=${OUTPUT_DIR}"

echo "[pipeline] converting dataset"
"${CONDA_BIN}" run -n lerobot-qrp312 python scripts/data_convert/convert_zeno_h1_v30.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${REPO_ROOT}/Data/lerobot" \
  --repo-name "${DATASET_REPO_ID}" \
  --task human_new_pick \
  --fps 20 \
  --img-size 224 \
  --overwrite

echo "[pipeline] conversion summary"
"${CONDA_BIN}" run -n lerobot-qrp312 python - <<'PY'
import json
from pathlib import Path

root = Path("/home/zeno-rp/2027icra/Data/lerobot/new_zeno_h1_v30")
info_path = root / "meta" / "info.json"
stats_path = root / "meta" / "stats.json"
with info_path.open("r", encoding="utf-8") as f:
    info = json.load(f)
print("dataset_root", root)
print("total_frames", info.get("total_frames"))
print("total_episodes", info.get("total_episodes"))
print("fps", info.get("fps"))
print("stats_exists", stats_path.is_file())
PY

echo "[pipeline] training ACT+DINOv2"
"${CONDA_BIN}" run -n lerobot-qrp312 env \
  DEVICE=cuda \
  USE_AMP=true \
  STEPS=100000 \
  BATCH_SIZE=8 \
  NUM_WORKERS=4 \
  SAVE_FREQ=20000 \
  LOG_FREQ=200 \
  DINOV2_PRETRAINED=true \
  DINOV2_TRAIN_BACKBONE=true \
  DATASET_REPO_ID="${DATASET_REPO_ID}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  JOB_NAME="${RUN_ID}" \
  TRAIN_LOG_DIR="${REPO_ROOT}/scripts/train_log/${RUN_ID}" \
  bash scripts/train_humanmoid_pick_act_dinov2.sh

echo "[pipeline] finished $(date --iso-8601=seconds)"
