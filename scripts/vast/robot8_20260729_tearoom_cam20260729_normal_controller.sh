#!/usr/bin/env bash
# Supervisor-owned normal-only TeaRoom pipeline:
# conversion -> durable frozen-DINO cache -> two-GPU DDP normal training.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260729_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/Data/lerobot/${SOURCE_REPO_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260729_tearoom_topcam_left_cam20260729_normal_dinov3_disk.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-18}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-37546}"
EXPECTED_TOPCAM_PROFILE="${EXPECTED_TOPCAM_PROFILE:-cam_20260729}"
EXPECTED_TOPCAM_PIPELINE="${EXPECTED_TOPCAM_PIPELINE:-split_left_right_then_cam_20260729_opencv_fisheye_rectify_then_independent_left_right_rgb_resize}"
EXPECTED_TOPCAM_RECTIFIED_HEIGHT="${EXPECTED_TOPCAM_RECTIFIED_HEIGHT:-720}"
EXPECTED_TOPCAM_CALIBRATION_SHA256="${EXPECTED_TOPCAM_CALIBRATION_SHA256:-}"
NORMAL_TRAIN_PROGRAM="${NORMAL_TRAIN_PROGRAM:-robot8_20260729_tearoom_cam20260729_normal_train}"
NORMAL_SUCCESS="${NORMAL_SUCCESS:-${REPO_ROOT}/outputs/train/robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp64_b32_60k/TRAINING_SUCCEEDED}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260729_cam20260729_rectified_convert.sh}"
CACHE_BUILD_SCRIPT="${CACHE_BUILD_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260729_tearoom_cam20260729_normal_cache.sh}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 1 )); then
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 1
fi
if ! [[ "${EXPECTED_EPISODES}" =~ ^[1-9][0-9]*$ && "${EXPECTED_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EXPECTED_EPISODES and EXPECTED_FRAMES must be positive decimal integers\n' >&2
  exit 1
fi

program_state() {
  local line
  line="$(supervisorctl status "$1" 2>/dev/null || true)"
  awk '{print $2}' <<<"${line}"
}

source_ready() {
  "${VENV_DIR}/bin/python" - \
    "${SOURCE_DATASET}" "${EXPECTED_TOPCAM_PROFILE}" "${EXPECTED_TOPCAM_PIPELINE}" \
    "${EXPECTED_TOPCAM_RECTIFIED_HEIGHT}" "${EXPECTED_TOPCAM_CALIBRATION_SHA256}" \
    "${EXPECTED_EPISODES}" "${EXPECTED_FRAMES}" <<'PY'
import json
import sys
from pathlib import Path

try:
    root = Path(sys.argv[1])
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    top = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
profile, pipeline = sys.argv[2], sys.argv[3]
rectified_height = int(sys.argv[4])
calibration_sha = sys.argv[5]
expected = {"fps": 20, "total_episodes": int(sys.argv[6]), "total_frames": int(sys.argv[7])}
if {key: info.get(key) for key in expected} != expected:
    raise SystemExit(1)
if top.get("profile") != profile or top.get("pipeline") != pipeline:
    raise SystemExit(1)
if top.get("rectified_size_per_eye") != {"width": 1280, "height": rectified_height}:
    raise SystemExit(1)
if top.get("spatial_crop") != {"x": 20, "y": 0, "width": 1240, "height": 620}:
    raise SystemExit(1)
if calibration_sha and top.get("calibration_sha256") != calibration_sha:
    raise SystemExit(1)
if top.get("head_camera_feature_to_eye") != {"head_cam": "left"}:
    raise SystemExit(1)
if top.get("selected_model_topcam_eye") != "left":
    raise SystemExit(1)
PY
}

cache_ready() {
  "${VENV_DIR}/bin/python" - \
    "${CACHE_MANIFEST}" "${SOURCE_REPO_ID}" "${SOURCE_DATASET}" \
    "${REQUIRE_TOPCAM_CACHE_PROVENANCE:-0}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if payload.get("storage") != "disk" or payload.get("ready") is not True or payload.get("status") != "ready":
        raise ValueError("cache is not ready")
    if payload.get("dataset", {}).get("repo_id") != sys.argv[2]:
        raise ValueError("wrong source dataset")
    cache = Path(payload["cache_path"])
    if cache.stat().st_size != int(payload["byte_size"]):
        raise ValueError("cache byte size mismatch")
    if sys.argv[4] == "1":
        digest = hashlib.sha256((Path(sys.argv[3]) / "meta" / "topcam_rectification.json").read_bytes()).hexdigest()
        if payload.get("topcam_rectification", {}).get("sha256") != digest:
            raise ValueError("topcam provenance mismatch")
except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
    raise SystemExit(1)
PY
}

if ! source_ready; then
  (
    export DATASET_REPO_ID="${SOURCE_REPO_ID}"
    export DATASET_ROOT="${SOURCE_DATASET}"
    bash "${CONVERT_SCRIPT}"
  )
fi
source_ready || { printf 'Source conversion did not produce the required TeaRoom cam contract\n' >&2; exit 1; }

if ! cache_ready; then
  bash "${CACHE_BUILD_SCRIPT}"
fi
cache_ready || { printf 'DINO cache did not produce the required TeaRoom source contract\n' >&2; exit 1; }

state="$(program_state "${NORMAL_TRAIN_PROGRAM}")"
case "${state}" in
  STOPPED)
    printf '[%s] starting %s\n' "$(date '+%F %T')" "${NORMAL_TRAIN_PROGRAM}"
    supervisorctl start "${NORMAL_TRAIN_PROGRAM}"
    ;;
  RUNNING|STARTING)
    printf '[%s] %s already %s\n' "$(date '+%F %T')" "${NORMAL_TRAIN_PROGRAM}" "${state}"
    ;;
  *)
    printf 'Cannot safely start %s because supervisor state is %s\n' \
      "${NORMAL_TRAIN_PROGRAM}" "${state:-unknown}" >&2
    exit 1
    ;;
esac

sleep 10
state="$(program_state "${NORMAL_TRAIN_PROGRAM}")"
if [[ "${state}" != RUNNING && "${state}" != STARTING ]]; then
  printf '%s failed to remain running after launch: %s\n' \
    "${NORMAL_TRAIN_PROGRAM}" "${state:-unknown}" >&2
  exit 1
fi
printf '[%s] two-GPU normal training launched; V3 intentionally deferred\n' "$(date '+%F %T')"

while true; do
  state="$(program_state "${NORMAL_TRAIN_PROGRAM}")"
  if [[ "${state}" == RUNNING || "${state}" == STARTING ]]; then
    printf '[%s] normal training state: %s\n' "$(date '+%F %T')" "${state}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  if [[ "${state}" == EXITED && -f "${NORMAL_SUCCESS}" ]]; then
    printf '[%s] normal training completed successfully\n' "$(date '+%F %T')"
    exit 0
  fi
  printf 'Normal training exited without its success sentinel. state=%s success=%s\n' \
    "${state:-unknown}" "$(test -f "${NORMAL_SUCCESS}" && printf yes || printf no)" >&2
  exit 1
done
