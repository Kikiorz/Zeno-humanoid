#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data/humanmoid_pick}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Data/lerobot}"
REPO_NAME="${REPO_NAME:-humanmoid_pick_zeno_h1_v30}"
TASK="${TASK:-humanmoid_pick}"
FPS="${FPS:-20}"
IMG_SIZE="${IMG_SIZE:-224}"

EXTRA_ARGS=()
if [[ "${INCLUDE_TORSO:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--include-torso)
fi

python3 "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --repo-name "${REPO_NAME}" \
  --task "${TASK}" \
  --fps "${FPS}" \
  --img-size "${IMG_SIZE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
