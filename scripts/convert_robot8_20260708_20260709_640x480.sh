#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV-lerobot-qrp312}"
if ! command -v conda >/dev/null 2>&1 && [[ "${CONDA_ENV}" == "lerobot-qrp312" ]]; then
  CONDA_ENV=""
fi
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Data/lerobot}"
CENTER_CROP_FRACTION="${CENTER_CROP_FRACTION:-0.6666667}"
ENCODER_THREADS="${ENCODER_THREADS:-8}"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

convert_one() {
  local data_dir="$1"
  local repo_name="$2"
  local task="$3"
  local cameras="$4"

  echo "[$(date '+%F %T')] converting ${repo_name}"
  echo "[$(date '+%F %T')] cameras=${cameras} data_dir=${data_dir}"

  runner=()
  if [[ -n "${CONDA_ENV}" ]]; then
    runner=(conda run --no-capture-output -n "${CONDA_ENV}")
  fi

  "${runner[@]}" \
    python3 "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
      --data-dir "${data_dir}" \
      --output-dir "${OUTPUT_DIR}" \
      --repo-name "${repo_name}" \
      --task "${task}" \
      --fps 20 \
      --img-width 640 \
      --img-height 480 \
      --center-crop-fraction "${CENTER_CROP_FRACTION}" \
      --cameras "${cameras}" \
      --vcodec h264 \
      --video-crf 18 \
      --video-gop 2 \
      --video-fast-decode 1 \
      --video-preset veryfast \
      --encoder-threads "${ENCODER_THREADS}" \
      --overwrite
}

convert_one \
  "${REPO_ROOT}/Data/2026_07_08" \
  "robot8_20260708_zeno_h1_auto_cmd_v30_center_crop_2of3_640x480" \
  "robot8_20260708" \
  "head_cam,left_arm_cam,right_arm_cam"

convert_one \
  "${REPO_ROOT}/Data/2026_07_09" \
  "robot8_20260709_zeno_h1_auto_cmd_v30_640x480_crop2of3_head_right" \
  "robot8_20260709" \
  "head_cam,right_arm_cam"

echo "[$(date '+%F %T')] all conversions finished"
