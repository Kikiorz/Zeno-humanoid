#!/usr/bin/env bash
# Convert the TeaRoom ROS2 bags into a three-camera LeRobot dataset using the
# calibrated 2026-07-29 fisheye rectification contract under data_convert/cam.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot-qrp312}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data/2026_7_29_tearoom}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Data/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260729_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/${DATASET_REPO_ID}}"
TASK_LABEL="${TASK_LABEL:-robot8_20260729_tearoom}"
CALIBRATION="${CALIBRATION:-${REPO_ROOT}/scripts/data_convert/cam/stereo_params_20260729_172611.npz}"
EXCLUDE_BAGS="${EXCLUDE_BAGS:-rosbag2_2026_07_29_19_59_37}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-18}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-37546}"

if [[ ! -d "${DATA_DIR}" ]]; then
  printf 'Raw TeaRoom data directory does not exist: %s\n' "${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CALIBRATION}" ]]; then
  printf 'Top-camera calibration does not exist: %s\n' "${CALIBRATION}" >&2
  exit 1
fi
if [[ -e "${DATASET_ROOT}" ]]; then
  printf 'Refusing to overwrite an existing dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi

runner=(python3)
if [[ -n "${CONDA_ENV}" ]]; then
  runner=(conda run --no-capture-output -n "${CONDA_ENV}" python3)
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${OUTPUT_ROOT}"

printf '[%s] converting %s to %s\n' "$(date '+%F %T')" "${DATA_DIR}" "${DATASET_ROOT}"
"${runner[@]}" "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --repo-name "${DATASET_REPO_ID}" \
  --task "${TASK_LABEL}" \
  --fps 20 \
  --img-width 640 --img-height 480 \
  --center-crop-fraction 1.0 \
  --rectify-head-stereo \
  --head-stereo-profile cam_20260729 \
  --head-stereo-cam-calibration "${CALIBRATION}" \
  --head-stereo-resize-mode letterbox \
  --cameras head_cam,left_arm_cam,right_arm_cam \
  --frozen-fields '' \
  --exclude-bags "${EXCLUDE_BAGS}" \
  --vcodec h264 --video-crf 18 --video-gop 2 --video-fast-decode 1 \
  --video-preset veryfast --encoder-threads 16

"${runner[@]}" - "${DATASET_ROOT}" "${CALIBRATION}" "${EXPECTED_EPISODES}" "${EXPECTED_FRAMES}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
calibration = Path(sys.argv[2])
expected_episodes = int(sys.argv[3])
expected_frames = int(sys.argv[4])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
expected_info = {
    "fps": 20,
    "total_episodes": expected_episodes,
    "total_frames": expected_frames,
}
actual_info = {key: info.get(key) for key in expected_info}
if actual_info != expected_info:
    raise SystemExit(f"Converted dataset metadata mismatch: {actual_info!r}")
for key in (
    "observation.images.head_cam",
    "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
):
    feature = info.get("features", {}).get(key, {})
    if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
        raise SystemExit(f"Invalid visual feature {key}: {feature!r}")
topcam = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
expected_topcam = {
    "profile": "cam_20260729",
    "pipeline": "split_left_right_then_cam_20260729_opencv_fisheye_rectify_then_independent_left_right_rgb_resize",
    "selected_model_topcam_eye": "left",
    "emitted_head_eyes": ["left"],
    "head_camera_feature_to_eye": {"head_cam": "left"},
    "head_stereo_model_resize_mode": "letterbox",
    "generic_center_crop_applied_to_head_stereo": False,
    "rectified_size_per_eye": {"width": 1280, "height": 720},
    "spatial_crop": {"x": 20, "y": 0, "width": 1240, "height": 620},
    "cropped_size_per_eye": {"width": 1240, "height": 620},
}
for key, expected in expected_topcam.items():
    if topcam.get(key) != expected:
        raise SystemExit(f"Top-camera provenance mismatch for {key}: {topcam.get(key)!r}")
expected_hash = hashlib.sha256(calibration.read_bytes()).hexdigest()
if topcam.get("calibration_sha256") != expected_hash:
    raise SystemExit("Top-camera calibration hash does not match the requested NPZ")
print(
    f"Verified {root}: {expected_episodes} episodes, {expected_frames} frames, "
    "cam_20260729 rectification.",
    flush=True,
)
PY
