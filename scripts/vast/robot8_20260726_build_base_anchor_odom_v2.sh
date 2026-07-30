#!/usr/bin/env bash
# Build a separate endpoint-anchored base-trajectory dataset for 2026-07-26.
# It never modifies the all-23D source dataset; videos are hard-linked only
# after every trajectory and endpoint validation has passed.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/Data/lerobot/robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23}"
BAG_DATA_DIR="${BAG_DATA_DIR:-${REPO_ROOT}/Data/2026_07_26}"
OUTPUT_DATASET="${OUTPUT_DATASET:-${REPO_ROOT}/Data/lerobot/robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23_base_anchor_odom_v2}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/outputs/analysis/robot8_20260726_base_anchor_odom_v2}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_DATASET}" || ! -d "${BAG_DATA_DIR}" ]]; then
  printf 'Missing source dataset or raw bags.\n' >&2
  exit 1
fi
if [[ -e "${OUTPUT_DATASET}" ]]; then
  printf 'Refusing to overwrite derived V2 dataset: %s\n' "${OUTPUT_DATASET}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_clean/reconstruct_robot8_20260721_base_anchor_odom_v2.py" \
  --source-dataset "${SOURCE_DATASET}" \
  --bag-data-dir "${BAG_DATA_DIR}" \
  --output-dataset "${OUTPUT_DATASET}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --preview-episodes 3 \
  --video-mode hardlink
