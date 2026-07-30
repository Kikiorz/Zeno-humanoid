#!/usr/bin/env bash
# Read-only acceptance audit for the V3 endpoint-constrained de-jitter labels.
#
# V3 deliberately smooths translation and yaw independently without forcing
# genuinely simultaneous holonomic motion into an image-inconsistent sequence.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/Data/2026_07_26}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_ROOT}/${DATASET_REPO_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/outputs/analysis/robot8_20260726_base_anchor_odom_v3_decoupled_smooth_dryrun}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_DATASET}" || ! -d "${RAW_DATA_DIR}" ]]; then
  printf 'Missing source dataset or raw bags. dataset=%s bags=%s\n' \
    "${SOURCE_DATASET}" "${RAW_DATA_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "${ANALYSIS_DIR}"
exec "${VENV_DIR}/bin/python" \
  "${REPO_ROOT}/scripts/data_clean/reconstruct_robot8_20260721_base_anchor_odom_v2.py" \
  --source-dataset "${SOURCE_DATASET}" \
  --bag-data-dir "${RAW_DATA_DIR}" \
  --output-dataset "${DATASET_ROOT}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --preview-episodes 3 \
  --dry-run \
  --dataset-version base_anchor_odom_v3_decoupled_smooth \
  --metadata-dir-name base_anchor_odom_v3_decoupled_smooth \
  --secondary-smooth-window 11
