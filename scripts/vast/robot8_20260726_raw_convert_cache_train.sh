#!/usr/bin/env bash
# Convert the validated 2026-07-26 ROS bags and train the untouched-label
# DINOv3/ACT baseline.  This intentionally never overwrites a dataset or run:
# an interrupted conversion must be inspected/removed deliberately rather than
# accidentally becoming a training input.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data/2026_07_26}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Data/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_decoder7_b32x2_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  printf 'Missing raw 2026-07-26 bags: %s\n' "${DATA_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite training output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
if [[ ! -e "${DATASET_ROOT}" ]]; then
  convert_log="${LOG_DIR}/convert_robot8_20260726_20hz_640x480_all23.log"
  printf '[%s] starting conversion: %s -> %s\n' \
    "$(date '+%F %T')" "${DATA_DIR}" "${DATASET_ROOT}" | tee -a "${convert_log}"
  "${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}" \
    --repo-name "${DATASET_REPO_ID}" \
    --task robot8_20260726 \
    --fps 20 \
    --img-width 640 --img-height 480 \
    --center-crop-fraction 1.0 \
    --cameras head_cam,left_arm_cam,right_arm_cam \
    --frozen-fields '' \
    --vcodec h264 --video-crf 18 --video-gop 2 --video-fast-decode 1 \
    --video-preset veryfast --encoder-threads 16 \
    2>&1 | tee -a "${convert_log}"
else
  printf '[%s] reusing completed dataset: %s\n' "$(date '+%F %T')" "${DATASET_ROOT}"
fi

# A failed converter leaves its output directory behind.  Do not let a later
# supervisor restart treat that partial directory as a valid cache/training
# input.  The 20 bags passed raw preflight with exactly 44,490 common 20 Hz
# samples, so these inexpensive metadata checks are a useful hard boundary.
"${VENV_DIR}/bin/python" - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info_path = root / "meta" / "info.json"
if not info_path.is_file():
    raise SystemExit(f"converted dataset is missing {info_path}")
info = json.loads(info_path.read_text(encoding="utf-8"))
expected = {"fps": 20, "total_episodes": 20, "total_frames": 44_490}
actual = {key: info.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"converted dataset metadata mismatch: expected={expected}, actual={actual}")
for key in ("observation.state", "action"):
    feature = info.get("features", {}).get(key, {})
    if feature.get("shape") != [23]:
        raise SystemExit(f"{key} must be 23-D, got {feature.get('shape')!r}")
print(f"conversion metadata verified: {actual}", flush=True)
PY

# The cache wrapper precomputes the frozen backbone with TorchCodec, retains
# its memfd for the whole run, then starts ACT.  32 is per GPU on this 2-GPU
# host, so the effective global batch is 64.
export RUN_ID DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR VENV_DIR
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
# This is only the one-off frozen-backbone cache batch, not ACT's training
# batch.  The 5090 preflight passed 64 samples/GPU comfortably; using it cuts
# cache scheduler overhead while the requested training batch remains 32/GPU.
export DINO_CACHE_BATCH_SIZE="${DINO_CACHE_BATCH_SIZE:-64}"
export DINO_CACHE_MIN_FREE_GIB="${DINO_CACHE_MIN_FREE_GIB:-64}"
export DINO_CACHE_VIDEO_BACKEND="${DINO_CACHE_VIDEO_BACKEND:-torchcodec}"
export ACTION_LOSS_WEIGHTS='[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'

exec bash "${REPO_ROOT}/scripts/run_robot8_20260721_act_dinov3_ddp128_all23_cached.sh"
