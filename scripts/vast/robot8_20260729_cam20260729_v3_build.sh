#!/usr/bin/env bash
# Construct the endpoint-anchored / decoupled-smooth V3 labels without
# modifying the cam_20260729 rectified visual source.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/Data/2026_07_29}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260729_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_ROOT}/${DATASET_REPO_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/outputs/analysis/robot8_20260729_cam20260729_base_anchor_odom_v3_decoupled_smooth}"
EXPECTED_TOPCAM_PROFILE="${EXPECTED_TOPCAM_PROFILE:-cam_20260729}"
EXPECTED_TOPCAM_PIPELINE="${EXPECTED_TOPCAM_PIPELINE:-split_left_right_then_cam_20260729_opencv_fisheye_rectify_then_independent_left_right_rgb_resize}"
EXPECTED_TOPCAM_RECTIFIED_HEIGHT="${EXPECTED_TOPCAM_RECTIFIED_HEIGHT:-720}"
EXPECTED_TOPCAM_CALIBRATION_SHA256="${EXPECTED_TOPCAM_CALIBRATION_SHA256:-}"
EXPECTED_TOPCAM_PROCESSING_SHA256="${EXPECTED_TOPCAM_PROCESSING_SHA256:-}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" || ! -d "${SOURCE_DATASET}" ]]; then
  printf 'Missing raw bags or rectified source. bags=%s source=%s\n' \
    "${RAW_DATA_DIR}" "${SOURCE_DATASET}" >&2
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
mkdir -p "${ANALYSIS_DIR}"

"${VENV_DIR}/bin/python" - \
  "${SOURCE_DATASET}" "${EXPECTED_TOPCAM_PROFILE}" "${EXPECTED_TOPCAM_PIPELINE}" \
  "${EXPECTED_TOPCAM_RECTIFIED_HEIGHT}" "${EXPECTED_TOPCAM_CALIBRATION_SHA256}" \
  "${EXPECTED_TOPCAM_PROCESSING_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

meta = Path(sys.argv[1]) / "meta" / "topcam_rectification.json"
payload = json.loads(meta.read_text(encoding="utf-8"))
profile, pipeline = sys.argv[2], sys.argv[3]
rectified_height = int(sys.argv[4])
calibration_sha, processing_sha = sys.argv[5], sys.argv[6]
if payload.get("profile") != profile:
    raise SystemExit(f"source uses the wrong topcam profile: {payload.get('profile')!r}")
if payload.get("pipeline") != pipeline:
    raise SystemExit(f"source uses the wrong topcam pipeline: {payload.get('pipeline')!r}")
if payload.get("rectified_size_per_eye") != {"width": 1280, "height": rectified_height}:
    raise SystemExit(f"source uses wrong rectified size: {payload.get('rectified_size_per_eye')!r}")
if calibration_sha and payload.get("calibration_sha256") != calibration_sha:
    raise SystemExit("source uses wrong topcam calibration hash")
if processing_sha and payload.get("processing_sha256") != processing_sha:
    raise SystemExit("source uses wrong topcam processing hash")
if payload.get("head_camera_feature_to_eye") != {"head_cam": "left"}:
    raise SystemExit("source has incorrect head-camera feature mapping")
if payload.get("selected_model_topcam_eye") != "left":
    raise SystemExit("source must retain only the aligned rectified left topcam eye")
if payload.get("head_stereo_model_resize_mode") != "letterbox" or payload.get("spatial_crop") != {
    "x": 20, "y": 0, "width": 1240, "height": 620
}:
    raise SystemExit("source violates the fixed-crop left-topcam letterbox contract")
PY

printf '[%s] deriving endpoint-accurate V3 labels: %s\n' "$(date '+%F %T')" "${DATASET_ROOT}"
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
