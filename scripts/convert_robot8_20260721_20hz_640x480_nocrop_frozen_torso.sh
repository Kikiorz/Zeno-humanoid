#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot-qrp312}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data/2026_07_21}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Data/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_frozen_lift_waist}"
TASK="${TASK:-robot8_20260721}"
CAMERAS="${CAMERAS:-head_cam,left_arm_cam,right_arm_cam}"
FROZEN_FIELDS="${FROZEN_FIELDS:-torso_lift,torso_waist}"
ENCODER_THREADS="${ENCODER_THREADS:-8}"
OVERWRITE="${OVERWRITE:-false}"

# These interrupted recordings contain zero-byte MCAP files and no metadata.
EXCLUDE_BAGS="${EXCLUDE_BAGS:-rosbag2_2026_07_21_15_09_46,rosbag2_2026_07_21_16_11_11}"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

runner=()
if [[ -n "${CONDA_ENV}" ]]; then
  runner=(conda run --no-capture-output -n "${CONDA_ENV}")
fi

cmd=(
  "${runner[@]}"
  python3 "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_ROOT}"
  --repo-name "${DATASET_REPO_ID}"
  --task "${TASK}"
  --fps 20
  --img-width 640
  --img-height 480
  --center-crop-fraction 1.0
  --cameras "${CAMERAS}"
  --frozen-fields "${FROZEN_FIELDS}"
  --vcodec h264
  --video-crf 18
  --video-gop 2
  --video-fast-decode 1
  --video-preset veryfast
  --encoder-threads "${ENCODER_THREADS}"
  --exclude-bags "${EXCLUDE_BAGS}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  cmd+=(--overwrite)
fi

log_dir="${REPO_ROOT}/outputs/logs"
log_path="${log_dir}/convert_robot8_20260721_20hz_640x480_nocrop_frozen_torso.log"
mkdir -p "${log_dir}"

printf '[%s] dataset=%s\n' "$(date '+%F %T')" "${DATASET_REPO_ID}"
printf '[%s] output=%s\n' "$(date '+%F %T')" "${OUTPUT_ROOT}/${DATASET_REPO_ID}"
printf '[%s] frozen=%s\n' "$(date '+%F %T')" "${FROZEN_FIELDS}"
printf '%q ' "${cmd[@]}" | tee "${log_path}"
printf '\n' | tee -a "${log_path}"

"${cmd[@]}" "$@" 2>&1 | tee -a "${log_path}"
