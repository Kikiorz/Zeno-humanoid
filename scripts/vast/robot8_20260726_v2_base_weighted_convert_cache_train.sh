#!/usr/bin/env bash
# End-to-end launcher for the independent, endpoint-anchored Robot8 V2 run.
#
# This intentionally keeps the original 2026-07-26 dataset untouched.  It
# first creates (or verifies) the raw-label 20 Hz dataset, then derives a
# physically anchored V2 copy, precomputes frozen DINOv3 maps, and finally
# trains the 7-decoder-layer ACT policy with extra base-action loss weight.
# It is intended for a separate dual-GPU host, so it never competes with the
# raw-label baseline's anonymous RAM DINO cache.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/Data/2026_07_26}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23_base_anchor_odom_v2}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_ROOT}/${DATASET_REPO_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/outputs/analysis/robot8_20260726_base_anchor_odom_v2}"
RUN_ID="${RUN_ID:-robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_base_anchor_odom_v2_base3x_decoder7_b32x2_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" ]]; then
  printf 'Missing raw 2026-07-26 bags: %s\n' "${RAW_DATA_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite V2 training output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${LOG_DIR}" "${LEROBOT_ROOT}"

if [[ ! -e "${SOURCE_DATASET}" ]]; then
  convert_log="${LOG_DIR}/convert_robot8_20260726_20hz_640x480_v2_source.log"
  printf '[%s] creating V2 source dataset: %s -> %s\n' \
    "$(date '+%F %T')" "${RAW_DATA_DIR}" "${SOURCE_DATASET}" | tee -a "${convert_log}"
  "${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir "${LEROBOT_ROOT}" \
    --repo-name "${SOURCE_REPO_ID}" \
    --task robot8_20260726 \
    --fps 20 \
    --img-width 640 --img-height 480 \
    --center-crop-fraction 1.0 \
    --cameras head_cam,left_arm_cam,right_arm_cam \
    --frozen-fields '' \
    --vcodec h264 --video-crf 18 --video-gop 2 --video-fast-decode 1 \
    --video-preset veryfast --encoder-threads 16 \
    2>&1 | tee -a "${convert_log}"
fi

# Do not silently consume a partial conversion after an interrupted transfer.
"${VENV_DIR}/bin/python" - "${SOURCE_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info_path = root / "meta" / "info.json"
if not info_path.is_file():
    raise SystemExit(f"source dataset is missing {info_path}")
info = json.loads(info_path.read_text(encoding="utf-8"))
expected = {"fps": 20, "total_episodes": 20, "total_frames": 44_490}
actual = {key: info.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"source dataset metadata mismatch: expected={expected}, actual={actual}")
for key in ("observation.state", "action"):
    if info.get("features", {}).get(key, {}).get("shape") != [23]:
        raise SystemExit(f"source {key} must be 23-D")
print(f"V2 source metadata verified: {actual}")
PY

if [[ ! -e "${DATASET_ROOT}" ]]; then
  printf '[%s] building endpoint-anchored V2 labels: %s\n' \
    "$(date '+%F %T')" "${DATASET_ROOT}" | tee -a "${LOG_DIR}/build_robot8_20260726_v2.log"
  "${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_clean/reconstruct_robot8_20260721_base_anchor_odom_v2.py" \
    --source-dataset "${SOURCE_DATASET}" \
    --bag-data-dir "${RAW_DATA_DIR}" \
    --output-dataset "${DATASET_ROOT}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --preview-episodes 3 \
    --video-mode hardlink \
    2>&1 | tee -a "${LOG_DIR}/build_robot8_20260726_v2.log"
fi

# A V2 action tail is desired physical twist, not raw /twist/cmd.  Demand the
# mapper and the fully written summary before allocating a 229 GiB feature
# cache or starting training.
"${VENV_DIR}/bin/python" - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
expected = {"fps": 20, "total_episodes": 20, "total_frames": 44_490}
actual = {key: info.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"V2 metadata mismatch: expected={expected}, actual={actual}")
meta = root / "meta" / "base_anchor_odom_v2"
mapper = meta / "dynamics_feedback_mapper.json"
summary = meta / "generation_summary.json"
if not mapper.is_file() or not summary.is_file():
    raise SystemExit("V2 dataset is missing its deployment mapper or generation summary")
payload = json.loads(summary.read_text(encoding="utf-8"))
if int(payload.get("frames", -1)) != 44_490:
    raise SystemExit(f"V2 summary frame count mismatch: {payload.get('frames')!r}")
print(f"V2 metadata verified: {actual}; accepted segments={payload.get('accepted_replan_segments')}")
PY

export RUN_ID DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR VENV_DIR
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
# Requested batch is per GPU, hence global 64 for a two-GPU host.
export BATCH_SIZE="${BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DINO_CACHE_BATCH_SIZE="${DINO_CACHE_BATCH_SIZE:-64}"
export DINO_CACHE_MIN_FREE_GIB="${DINO_CACHE_MIN_FREE_GIB:-64}"
export DINO_CACHE_VIDEO_BACKEND="${DINO_CACHE_VIDEO_BACKEND:-torchcodec}"
# The three base components are each 3×; combined base supervision is roughly
# comparable to the twenty upper-body action dimensions.
export ACTION_LOSS_WEIGHTS='[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3]'

exec bash "${REPO_ROOT}/scripts/run_robot8_20260721_act_dinov3_ddp128_all23_cached.sh"
