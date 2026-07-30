#!/usr/bin/env bash
# Consumer-only V3 training with the requested operation -> base -> equal
# objective curriculum. It reads the same visual memfd as the raw run.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_4cam_640x480_headstereo_rectified_crop_lr_all23}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260726_act_dinov3_4cam_640x480_headstereo_rectified_crop_lr_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260726_headstereo_rectified_lr_shared_dinov3.json}"

if [[ ! -d "${DATASET_ROOT}" || ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing corrected V3 dataset or environment. dataset=%s venv=%s\n' "${DATASET_ROOT}" "${VENV_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite V3 training output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi

"${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("ready") is not True or payload.get("status") != "ready":
    raise SystemExit(f"shared DINO cache is not ready: {payload.get('status')!r}")
pid = int(payload.get("parent_pid", -1))
try:
    os.kill(pid, 0)
except OSError as exc:
    raise SystemExit(f"shared DINO cache owner is not alive (pid={pid}): {exc}") from exc
cache = Path(str(payload["cache_path"]))
if cache.stat().st_size != int(payload["byte_size"]):
    raise SystemExit("shared DINO cache byte size changed")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export REPO_ROOT VENV_DIR DATASET_REPO_ID DATASET_ROOT RUN_ID OUTPUT_DIR
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export NUM_PROCESSES=1
export BATCH_SIZE=32
export GLOBAL_BATCH_SIZE=32
export NUM_WORKERS="${NUM_WORKERS:-8}"
export VIDEO_BACKEND=torchcodec
export DINOV2_TRAIN_BACKBONE=false
export DINO_FEATURE_CACHE_MANIFEST="${CACHE_MANIFEST}"
export ACTION_LOSS_WEIGHT_SCHEDULE_STEPS='[0,25000,60000,75000,90000,100000]'
export ACTION_LOSS_WEIGHT_SCHEDULE_VALUES="$(${VENV_DIR}/bin/python - <<'PY'
import json
groups = [(3.0, 1.0), (3.0, 1.0), (1.0, 3.0), (1.0, 3.0), (1.0, 1.0), (1.0, 1.0)]
print(json.dumps([[upper / 20.0] * 20 + [base / 3.0] * 3 for upper, base in groups], separators=(",", ":")))
PY
)"

exec bash "${REPO_ROOT}/scripts/train_robot8_20260721_act_dinov3_ddp128_all23_100k.sh"
