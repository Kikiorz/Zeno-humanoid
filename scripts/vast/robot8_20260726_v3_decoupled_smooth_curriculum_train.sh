#!/usr/bin/env bash
# Build the independent V3 de-jitter dataset, cache frozen DINOv3 maps, then
# train ACT with a base/upper-body loss curriculum.  The caller must keep this
# one foreground process group alive: the anonymous-RAM DINO cache is owned by
# the process and intentionally disappears when the job ends.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/Data/2026_07_26}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_ROOT}/${DATASET_REPO_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/outputs/analysis/robot8_20260726_base_anchor_odom_v3_decoupled_smooth}"
RUN_ID="${RUN_ID:-robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32x2_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"

# Use physical GPU 1 so GPU 0 remains wholly untouched for the user.  Inside
# this process group GPU 1 becomes logical cuda:0; cache-builder device ids
# below must therefore be relative to this visible set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" || ! -d "${SOURCE_DATASET}" ]]; then
  printf 'Missing raw bags or source data. bags=%s source=%s\n' "${RAW_DATA_DIR}" "${SOURCE_DATASET}" >&2
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
mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}"

if [[ ! -e "${DATASET_ROOT}" ]]; then
  build_log="${LOG_DIR}/build_robot8_20260726_v3_decoupled_smooth.log"
  printf '[%s] building V3 endpoint-anchored de-jitter labels: %s\n' \
    "$(date '+%F %T')" "${DATASET_ROOT}" | tee -a "${build_log}"
  "${VENV_DIR}/bin/python" \
    "${REPO_ROOT}/scripts/data_clean/reconstruct_robot8_20260721_base_anchor_odom_v2.py" \
    --source-dataset "${SOURCE_DATASET}" \
    --bag-data-dir "${RAW_DATA_DIR}" \
    --output-dataset "${DATASET_ROOT}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --preview-episodes 3 \
    --video-mode hardlink \
    --dataset-version base_anchor_odom_v3_decoupled_smooth \
    --metadata-dir-name base_anchor_odom_v3_decoupled_smooth \
    --secondary-smooth-window 11 \
    2>&1 | tee -a "${build_log}"
fi

# Refuse to spend roughly 229 GiB of RAM on DINO features unless the derived
# data is the audited V3 semantic and the second pass materially changed it.
"${VENV_DIR}/bin/python" - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"fps": 20, "total_episodes": 20, "total_frames": 44_490}
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
actual = {key: info.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"V3 metadata mismatch: expected={expected}, actual={actual}")
meta = root / "meta" / "base_anchor_odom_v3_decoupled_smooth"
for name in ("manifest.json", "generation_summary.json", "dynamics_feedback_mapper.json", "action_provenance.parquet"):
    if not (meta / name).is_file():
        raise SystemExit(f"V3 dataset is missing {meta / name}")
summary = json.loads((meta / "generation_summary.json").read_text(encoding="utf-8"))
if summary.get("version") != "base_anchor_odom_v3_decoupled_smooth":
    raise SystemExit(f"unexpected V3 version: {summary.get('version')!r}")
secondary = summary.get("secondary_decoupled_smoothing", {})
fraction = float(secondary.get("accepted_action_frame_fraction", -1.0))
reduction = float(secondary.get("velocity_total_variation_reduction_vs_v2", -1.0))
if fraction < 0.70 or reduction < 0.20:
    raise SystemExit(
        "V3 smoothing acceptance/reduction below training contract: "
        f"accepted_fraction={fraction:.3f}, tv_reduction={reduction:.3f}"
    )
print(
    "V3 metadata verified: "
    f"{actual}; accepted_action_fraction={fraction:.3f}; tv_reduction_vs_v2={reduction:.3f}"
)
PY

export RUN_ID DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR VENV_DIR
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DINO_CACHE_BATCH_SIZE="${DINO_CACHE_BATCH_SIZE:-64}"
export DINO_CACHE_MIN_FREE_GIB="${DINO_CACHE_MIN_FREE_GIB:-64}"
export DINO_CACHE_VIDEO_BACKEND="${DINO_CACHE_VIDEO_BACKEND:-torchcodec}"
export DINO_CACHE_DEVICES="${DINO_CACHE_DEVICES:-0}"
export DINOV2_TRAIN_BACKBONE=false

# Values are *group-normalized*: the first twenty values sum to the requested
# upper-body group total, and the last three sum to the requested base total.
# Thus early (0–25k) is upper:base=3:1, 25–60k transitions to 1:3, 60–75k
# holds base focus, and 75–90k returns to equal weighting for the final 10k.
export ACTION_LOSS_WEIGHT_SCHEDULE_STEPS='[0,25000,60000,75000,90000,100000]'
export ACTION_LOSS_WEIGHT_SCHEDULE_VALUES="$(${VENV_DIR}/bin/python - <<'PY'
import json

groups = [(3.0, 1.0), (3.0, 1.0), (1.0, 3.0), (1.0, 3.0), (1.0, 1.0), (1.0, 1.0)]
print(json.dumps([[upper / 20.0] * 20 + [base / 3.0] * 3 for upper, base in groups], separators=(",", ":")))
PY
)"

exec bash "${REPO_ROOT}/scripts/run_robot8_20260721_act_dinov3_ddp128_all23_cached.sh"
