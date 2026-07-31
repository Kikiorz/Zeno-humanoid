#!/usr/bin/env bash
# Convert the 2026-07-30 TeaRoom bags with the calibrated left-topcam contract.
# The 30 Hz clock is the measured minimum camera rate rounded to a 5 Hz multiple.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/Data/2026_07_30_TeaRoom}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260730_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_ROOT}/${DATASET_REPO_ID}}"
STAGING_PARENT="${STAGING_PARENT:-${LEROBOT_ROOT}/.${DATASET_REPO_ID}.staging-$$}"
STAGED_DATASET="${STAGED_DATASET:-${STAGING_PARENT}/${DATASET_REPO_ID}}"
CALIBRATION="${CALIBRATION:-${REPO_ROOT}/scripts/data_convert/cam/stereo_params_20260729_172611.npz}"
FPS="${FPS:-30}"
EXPECTED_CALIBRATION_SHA256="6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342"
# This recording is retained in the raw upload, but its MCAP tail lacks the
# mandatory end magic and rosbags cannot safely replay it.  Keep source and
# V3 discovery identical by excluding it explicitly in both stages.
EXCLUDE_BAGS="${EXCLUDE_BAGS:-rosbag2_2026_07_30_20_23_35}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" || ! -f "${CALIBRATION}" ]]; then
  printf 'Missing raw data or topcam calibration. raw=%s calibration=%s\n' \
    "${RAW_DATA_DIR}" "${CALIBRATION}" >&2
  exit 1
fi
if [[ -e "${DATASET_ROOT}" ]]; then
  printf 'Refusing to overwrite an existing dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -e "${STAGING_PARENT}" ]]; then
  printf 'Refusing to reuse an existing conversion staging directory: %s\n' "${STAGING_PARENT}" >&2
  exit 1
fi
if [[ "${FPS}" != 30 ]]; then
  printf 'This run is validated for the measured 30 Hz camera rate, got FPS=%s\n' "${FPS}" >&2
  exit 1
fi

actual_calibration_sha256="$(sha256sum "${CALIBRATION}" | awk '{print $1}')"
if [[ "${actual_calibration_sha256}" != "${EXPECTED_CALIBRATION_SHA256}" ]]; then
  printf 'Refusing a topcam calibration outside scripts/data_convert/cam contract: %s\n' "${CALIBRATION}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${LEROBOT_ROOT}"
mkdir -p "${STAGING_PARENT}"

printf '[%s] converting raw TeaRoom data at %s Hz: %s\n' \
  "$(date '+%F %T')" "${FPS}" "${RAW_DATA_DIR}"
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
  --data-dir "${RAW_DATA_DIR}" \
  --output-dir "${STAGING_PARENT}" \
  --repo-name "${DATASET_REPO_ID}" \
  --task robot8_20260730_tearoom \
  --fps "${FPS}" \
  --img-width 640 --img-height 480 \
  --center-crop-fraction 1.0 \
  --rectify-head-stereo \
  --head-stereo-profile cam_20260729 \
  --head-stereo-cam-calibration "${CALIBRATION}" \
  --head-stereo-resize-mode letterbox \
  --cameras head_cam,left_arm_cam,right_arm_cam \
  --frozen-fields '' \
  --exclude-bags "${EXCLUDE_BAGS}" \
  --fail-on-dropped-images \
  --vcodec h264 --video-crf 18 --video-gop 2 --video-fast-decode 1 \
  --video-preset veryfast --encoder-threads 16

"${VENV_DIR}/bin/python" - "${STAGED_DATASET}" "${CALIBRATION}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
calibration = Path(sys.argv[2])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
if info.get("fps") != 30 or info.get("total_episodes") != 4 or int(info.get("total_frames", 0)) < 50_000:
    raise SystemExit(f"Invalid converted source metadata: {info.get('fps')=}, {info.get('total_episodes')=}, {info.get('total_frames')=}")
for key in (
    "observation.images.head_cam",
    "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
):
    feature = info.get("features", {}).get(key, {})
    if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
        raise SystemExit(f"Invalid visual feature {key}: {feature!r}")
topcam = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
expected = {
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
for key, value in expected.items():
    if topcam.get(key) != value:
        raise SystemExit(f"Top-camera provenance mismatch for {key}: {topcam.get(key)!r}")
if topcam.get("calibration_sha256") != hashlib.sha256(calibration.read_bytes()).hexdigest():
    raise SystemExit("Top-camera calibration hash does not match scripts/data_convert/cam")
print(
    f"Verified 30 Hz source dataset: episodes={info['total_episodes']}, frames={info['total_frames']}; "
    "raw stereo -> rectified/aligned -> cropped left RGB -> 640x480 letterbox.",
    flush=True,
)
PY

mv -- "${STAGED_DATASET}" "${DATASET_ROOT}"
rmdir -- "${STAGING_PARENT}"
printf '[%s] atomically published verified source dataset: %s\n' "$(date '+%F %T')" "${DATASET_ROOT}"
