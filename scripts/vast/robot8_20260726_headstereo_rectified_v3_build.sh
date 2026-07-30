#!/usr/bin/env bash
# Derive the independent V3 endpoint-anchored/smoothed labels from the newly
# rectified source.  It must never point at the older unrectified dataset.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/Data/2026_07_26}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_4cam_640x480_headstereo_rectified_crop_lr_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_ROOT}/${DATASET_REPO_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/outputs/analysis/robot8_20260726_headstereo_rectified_base_anchor_odom_v3_decoupled_smooth}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" || ! -d "${SOURCE_DATASET}" ]]; then
  printf 'Missing raw bags or corrected source data. bags=%s source=%s\n' "${RAW_DATA_DIR}" "${SOURCE_DATASET}" >&2
  exit 1
fi
if [[ -e "${DATASET_ROOT}" ]]; then
  printf 'Refusing to reuse or overwrite a possibly partial V3 dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${ANALYSIS_DIR}"

"${VENV_DIR}/bin/python" - "${SOURCE_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

meta = Path(sys.argv[1]) / "meta" / "topcam_rectification.json"
payload = json.loads(meta.read_text(encoding="utf-8"))
if payload.get("output_eyes") != ["left", "right"]:
    raise SystemExit(f"source does not preserve separate rectified stereo eyes: {meta}")
if payload.get("head_camera_feature_to_eye") != {"head_cam": "left", "head_cam_right": "right"}:
    raise SystemExit(f"source has incorrect head camera feature mapping: {meta}")
if payload.get("head_stereo_model_resize_mode") != "letterbox":
    raise SystemExit(f"source uses a non-canonical topcam resize mode: {meta}")
PY

printf '[%s] deriving corrected V3 labels: %s\n' "$(date '+%F %T')" "${DATASET_ROOT}"
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
  --secondary-smooth-window 11

"${VENV_DIR}/bin/python" - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "meta" / "base_anchor_odom_v3_decoupled_smooth" / "generation_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("version") != "base_anchor_odom_v3_decoupled_smooth":
    raise SystemExit(f"unexpected V3 version: {summary.get('version')!r}")
secondary = summary.get("secondary_decoupled_smoothing", {})
fraction = float(secondary.get("accepted_action_frame_fraction", -1.0))
reduction = float(secondary.get("velocity_total_variation_reduction_vs_v2", -1.0))
if fraction < 0.70 or reduction < 0.20:
    raise SystemExit(
        "V3 smoothing contract failed: "
        f"accepted_action_frame_fraction={fraction:.3f}, tv_reduction={reduction:.3f}"
    )
print(f"V3 smoothing verified: accepted={fraction:.3f}; tv_reduction={reduction:.3f}", flush=True)
PY

"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_clean/verify_shared_visual_dataset.py" \
  --source-dataset "${SOURCE_DATASET}" \
  --derived-dataset "${DATASET_ROOT}"
