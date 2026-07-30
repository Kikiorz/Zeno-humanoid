#!/usr/bin/env bash
# Build a fresh, visually-correct Robot8 source dataset. The source head topic
# is one JPEG with two unrectified fisheye eyes; the converter emits the
# independently calibrated left and right views as two model cameras.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data/2026_07_26}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Data/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_4cam_640x480_headstereo_rectified_crop_lr_all23}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/${DATASET_REPO_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  printf 'Missing raw bags: %s\n' "${DATA_DIR}" >&2
  exit 1
fi
if [[ -e "${DATASET_ROOT}" ]]; then
  printf 'Refusing to reuse or overwrite a possibly partial corrected dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import importlib.util
missing = [name for name in ("rosbags", "cv2", "datasets", "pyarrow") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Conversion environment is missing required packages: " + ", ".join(missing))
PY

printf '[%s] converting corrected head-stereo source dataset: %s\n' \
  "$(date '+%F %T')" "${DATASET_ROOT}"
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --repo-name "${DATASET_REPO_ID}" \
  --task robot8_20260726 \
  --fps 20 \
  --img-width 640 --img-height 480 \
  --center-crop-fraction 1.0 \
  --rectify-head-stereo \
  --head-stereo-resize-mode letterbox \
  --cameras head_cam,head_cam_right,left_arm_cam,right_arm_cam \
  --frozen-fields '' \
  --vcodec h264 --video-crf 18 --video-gop 2 --video-fast-decode 1 \
  --video-preset veryfast --encoder-threads 16

"${VENV_DIR}/bin/python" - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
expected = {"fps": 20, "total_episodes": 20, "total_frames": 44_490}
actual = {key: info.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"corrected dataset metadata mismatch: expected={expected}, actual={actual}")
for key in (
    "observation.images.head_cam",
    "observation.images.head_cam_right",
    "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
):
    feature = info.get("features", {}).get(key, {})
    if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
        raise SystemExit(f"invalid image feature {key}: {feature!r}")
topcam = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
required = {
    "pipeline": "split_left_right_then_opencv_fisheye_rectify_then_crop_then_independent_left_right_rgb_resize",
    "output_eyes": ["left", "right"],
    "head_stereo_model_resize_mode": "letterbox",
    "generic_center_crop_applied_to_head_stereo": False,
}
if {key: topcam.get(key) for key in required} != required:
    raise SystemExit(f"topcam provenance does not match corrected-training contract: {topcam}")
if topcam.get("head_camera_feature_to_eye") != {"head_cam": "left", "head_cam_right": "right"}:
    raise SystemExit(f"topcam feature mapping does not preserve independent eyes: {topcam}")
print(f"corrected source dataset verified: {actual}; independent left/right rectification enabled", flush=True)
PY
