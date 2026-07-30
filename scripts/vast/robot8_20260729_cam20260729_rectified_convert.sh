#!/usr/bin/env bash
# Fresh source conversion for the 2026-07-29 recordings.  The raw head topic
# is the unrectified left|right fisheye pair.  The defaults preserve the
# historical NPZ run, while TOPCAM_PROFILE=data_process_20260729 selects the
# user-supplied Data/process calibration/alignment contract exactly.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data/2026_07_29}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Data/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260729_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/${DATASET_REPO_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"
EXPECTED_EPISODES=18
EXPECTED_FRAMES=19074
TOPCAM_PROFILE="${TOPCAM_PROFILE:-cam_20260729}"
HEAD_STEREO_CALIBRATION="${HEAD_STEREO_CALIBRATION:-}"
HEAD_STEREO_PROCESSING="${HEAD_STEREO_PROCESSING:-}"
HEAD_STEREO_CAM_CALIBRATION="${HEAD_STEREO_CAM_CALIBRATION:-}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  printf 'Missing raw bags: %s\n' "${DATA_DIR}" >&2
  exit 1
fi
if [[ -e "${DATASET_ROOT}" ]]; then
  printf 'Refusing to reuse or overwrite a possibly partial source dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi

case "${TOPCAM_PROFILE}" in
  data_process_20260729)
    HEAD_STEREO_CALIBRATION="${HEAD_STEREO_CALIBRATION:-${REPO_ROOT}/Data/process/top_stereo_calibration_basalt_kb4_compat.json}"
    HEAD_STEREO_PROCESSING="${HEAD_STEREO_PROCESSING:-${REPO_ROOT}/Data/process/processing_metadata_centered_crop_1240x620.json}"
    for path in "${HEAD_STEREO_CALIBRATION}" "${HEAD_STEREO_PROCESSING}"; do
      if [[ ! -f "${path}" ]]; then
        printf 'Missing exact Data/process topcam contract file: %s\n' "${path}" >&2
        exit 1
      fi
    done
    HEAD_STEREO_ARGS=(
      --head-stereo-calibration "${HEAD_STEREO_CALIBRATION}"
      --head-stereo-processing "${HEAD_STEREO_PROCESSING}"
    )
    EXPECTED_PIPELINE="split_left_right_then_opencv_fisheye_rectify_then_crop_then_independent_left_right_rgb_resize"
    EXPECTED_RECTIFIED_HEIGHT=620
    ;;
  cam_20260729)
    HEAD_STEREO_CAM_CALIBRATION="${HEAD_STEREO_CAM_CALIBRATION:-${REPO_ROOT}/scripts/data_convert/cam/stereo_params_20260729_172611.npz}"
    if [[ ! -f "${HEAD_STEREO_CAM_CALIBRATION}" ]]; then
      printf 'Missing 2026-07-29 NPZ topcam calibration: %s\n' "${HEAD_STEREO_CAM_CALIBRATION}" >&2
      exit 1
    fi
    HEAD_STEREO_ARGS=(--head-stereo-cam-calibration "${HEAD_STEREO_CAM_CALIBRATION}")
    EXPECTED_PIPELINE="split_left_right_then_cam_20260729_opencv_fisheye_rectify_then_independent_left_right_rgb_resize"
    EXPECTED_RECTIFIED_HEIGHT=720
    ;;
  *)
    printf 'Unsupported TOPCAM_PROFILE for this fresh 2026-07-29 conversion: %s\n' "${TOPCAM_PROFILE}" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import importlib.util
missing = [name for name in ("rosbags", "cv2", "datasets", "pyarrow") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Conversion environment is missing required packages: " + ", ".join(missing))
PY

printf '[%s] converting fresh 2026-07-29 %s head-stereo source: %s\n' \
  "$(date '+%F %T')" "${TOPCAM_PROFILE}" "${DATASET_ROOT}"
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --repo-name "${DATASET_REPO_ID}" \
  --task robot8_20260729 \
  --fps 20 \
  --img-width 640 --img-height 480 \
  --center-crop-fraction 1.0 \
  --rectify-head-stereo \
  --head-stereo-profile "${TOPCAM_PROFILE}" \
  "${HEAD_STEREO_ARGS[@]}" \
  --head-stereo-resize-mode letterbox \
  --cameras head_cam,left_arm_cam,right_arm_cam \
  --frozen-fields '' \
  --vcodec h264 --video-crf 18 --video-gop 2 --video-fast-decode 1 \
  --video-preset veryfast --encoder-threads 16

"${VENV_DIR}/bin/python" - \
  "${DATASET_ROOT}" "${EXPECTED_EPISODES}" "${EXPECTED_FRAMES}" \
  "${TOPCAM_PROFILE}" "${EXPECTED_PIPELINE}" "${EXPECTED_RECTIFIED_HEIGHT}" \
  "${HEAD_STEREO_CALIBRATION}" "${HEAD_STEREO_PROCESSING}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"fps": 20, "total_episodes": int(sys.argv[2]), "total_frames": int(sys.argv[3])}
profile = sys.argv[4]
pipeline = sys.argv[5]
rectified_height = int(sys.argv[6])
calibration = Path(sys.argv[7]) if sys.argv[7] else None
processing = Path(sys.argv[8]) if sys.argv[8] else None
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
actual = {key: info.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"source metadata mismatch: expected={expected}, actual={actual}")
for key in (
    "observation.images.head_cam",
    "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
):
    feature = info.get("features", {}).get(key, {})
    if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
        raise SystemExit(f"invalid visual feature {key}: {feature!r}")
topcam = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
required = {
    "profile": profile,
    "pipeline": pipeline,
    "output_eyes": ["left"],
    "spatial_crop": {"x": 20, "y": 0, "width": 1240, "height": 620},
    "head_stereo_model_resize_mode": "letterbox",
    "generic_center_crop_applied_to_head_stereo": False,
}
if {key: topcam.get(key) for key in required} != required:
    raise SystemExit(f"topcam provenance does not match {profile} contract: {topcam}")
if topcam.get("head_camera_feature_to_eye") != {"head_cam": "left"}:
    raise SystemExit(f"head eye mapping is invalid: {topcam}")
if topcam.get("selected_model_topcam_eye") != "left":
    raise SystemExit(f"source must retain only the rectified left topcam eye: {topcam}")
if topcam.get("emitted_head_eyes") != ["left"]:
    raise SystemExit(f"source must emit only the rectified left topcam eye: {topcam}")
if topcam.get("rectified_size_per_eye") != {"width": 1280, "height": rectified_height}:
    raise SystemExit(f"rectified size is invalid: {topcam.get('rectified_size_per_eye')!r}")
if topcam.get("cropped_size_per_eye") != {"width": 1240, "height": 620}:
    raise SystemExit(f"fixed crop size is invalid: {topcam.get('cropped_size_per_eye')!r}")
if profile == "data_process_20260729":
    if topcam.get("contract_source") != "Data/process/rectify_topcam_stereo.py":
        raise SystemExit(f"Data/process source marker missing: {topcam}")
    if calibration is None or processing is None:
        raise SystemExit("Data/process calibration/processing paths were not passed")
    expected_hashes = {
        "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "processing_sha256": hashlib.sha256(processing.read_bytes()).hexdigest(),
    }
    if {key: topcam.get(key) for key in expected_hashes} != expected_hashes:
        raise SystemExit(f"Data/process hash provenance mismatch: {topcam}")
print(f"{profile} source dataset verified: {actual}", flush=True)
PY
