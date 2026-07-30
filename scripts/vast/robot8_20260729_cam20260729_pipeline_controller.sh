#!/usr/bin/env bash
# Supervisor-owned orchestration of conversion -> V3 -> durable DINO cache ->
# two independent 100k trainings. The disk cache remains valid after this
# controller exits; it is intentionally never stopped or invalidated here.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260729_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/Data/lerobot/${SOURCE_REPO_ID}}"
V3_REPO_ID="${V3_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
V3_DATASET="${V3_DATASET:-${REPO_ROOT}/Data/lerobot/${V3_REPO_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260729_topcam_left_cam20260729_shared_dinov3_disk.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPECTED_TOPCAM_PROFILE="${EXPECTED_TOPCAM_PROFILE:-cam_20260729}"
EXPECTED_TOPCAM_PIPELINE="${EXPECTED_TOPCAM_PIPELINE:-split_left_right_then_cam_20260729_opencv_fisheye_rectify_then_independent_left_right_rgb_resize}"
EXPECTED_TOPCAM_RECTIFIED_HEIGHT="${EXPECTED_TOPCAM_RECTIFIED_HEIGHT:-720}"
EXPECTED_TOPCAM_CALIBRATION_SHA256="${EXPECTED_TOPCAM_CALIBRATION_SHA256:-}"
EXPECTED_TOPCAM_PROCESSING_SHA256="${EXPECTED_TOPCAM_PROCESSING_SHA256:-}"

RAW_TRAIN_PROGRAM="${RAW_TRAIN_PROGRAM:-robot8_20260729_cam20260729_raw_train}"
V3_TRAIN_PROGRAM="${V3_TRAIN_PROGRAM:-robot8_20260729_cam20260729_v3_train}"
RAW_SUCCESS="${RAW_SUCCESS:-${REPO_ROOT}/outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_b32_100k/TRAINING_SUCCEEDED}"
V3_SUCCESS="${V3_SUCCESS:-${REPO_ROOT}/outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k/TRAINING_SUCCEEDED}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260729_cam20260729_rectified_convert.sh}"
V3_BUILD_SCRIPT="${V3_BUILD_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260729_cam20260729_v3_build.sh}"
CACHE_BUILD_SCRIPT="${CACHE_BUILD_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260729_cam20260729_shared_dino_disk_cache.sh}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 1 )); then
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 1
fi

program_state() {
  local line
  line="$(supervisorctl status "$1" 2>/dev/null || true)"
  awk '{print $2}' <<<"${line}"
}

start_if_stopped() {
  local program="$1"
  local state
  state="$(program_state "${program}")"
  case "${state}" in
    STOPPED)
      printf '[%s] starting %s\n' "$(date '+%F %T')" "${program}"
      supervisorctl start "${program}"
      ;;
    RUNNING|STARTING)
      printf '[%s] %s already %s\n' "$(date '+%F %T')" "${program}" "${state}"
      ;;
    *)
      printf 'Cannot safely start %s because supervisor state is %s\n' \
        "${program}" "${state:-unknown}" >&2
      return 1
      ;;
  esac
}

source_ready() {
  "${VENV_DIR}/bin/python" - \
    "${SOURCE_DATASET}" "${EXPECTED_TOPCAM_PROFILE}" "${EXPECTED_TOPCAM_PIPELINE}" \
    "${EXPECTED_TOPCAM_RECTIFIED_HEIGHT}" "${EXPECTED_TOPCAM_CALIBRATION_SHA256}" \
    "${EXPECTED_TOPCAM_PROCESSING_SHA256}" <<'PY'
import json
import sys
from pathlib import Path
try:
    root = Path(sys.argv[1])
    info = json.loads((root / "meta" / "info.json").read_text())
    top = json.loads((root / "meta" / "topcam_rectification.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
profile, pipeline = sys.argv[2], sys.argv[3]
rectified_height = int(sys.argv[4])
calibration_sha, processing_sha = sys.argv[5], sys.argv[6]
if {k: info.get(k) for k in ("fps", "total_episodes", "total_frames")} != {"fps": 20, "total_episodes": 18, "total_frames": 19074}:
    raise SystemExit(1)
if top.get("profile") != profile or top.get("pipeline") != pipeline or top.get("spatial_crop") != {
    "x": 20, "y": 0, "width": 1240, "height": 620
}:
    raise SystemExit(1)
if top.get("rectified_size_per_eye") != {"width": 1280, "height": rectified_height}:
    raise SystemExit(1)
if calibration_sha and top.get("calibration_sha256") != calibration_sha:
    raise SystemExit(1)
if processing_sha and top.get("processing_sha256") != processing_sha:
    raise SystemExit(1)
if top.get("head_camera_feature_to_eye") != {"head_cam": "left"}:
    raise SystemExit(1)
if top.get("selected_model_topcam_eye") != "left":
    raise SystemExit(1)
PY
}

v3_ready() {
  "${VENV_DIR}/bin/python" - "${V3_DATASET}" <<'PY'
import json
import sys
from pathlib import Path
try:
    root = Path(sys.argv[1])
    summary = json.loads((root / "meta" / "base_anchor_odom_v3_decoupled_smooth" / "generation_summary.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
secondary = summary.get("secondary_decoupled_smoothing", {})
if summary.get("version") != "base_anchor_odom_v3_decoupled_smooth":
    raise SystemExit(1)
if float(secondary.get("accepted_action_frame_fraction", -1.0)) < 0.70:
    raise SystemExit(1)
if float(secondary.get("velocity_total_variation_reduction_vs_v2", -1.0)) < 0.20:
    raise SystemExit(1)
PY
  if [[ $? -ne 0 ]]; then
    return 1
  fi
  "${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_clean/verify_shared_visual_dataset.py" \
    --source-dataset "${SOURCE_DATASET}" \
    --derived-dataset "${V3_DATASET}"
}

cache_ready() {
  "${VENV_DIR}/bin/python" - \
    "${CACHE_MANIFEST}" "${SOURCE_REPO_ID}" "${SOURCE_DATASET}" \
    "${REQUIRE_TOPCAM_CACHE_PROVENANCE:-0}" "${REPO_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text())
    if payload.get("storage") != "disk" or payload.get("ready") is not True or payload.get("status") != "ready":
        raise ValueError("not ready")
    if payload.get("dataset", {}).get("repo_id") != sys.argv[2]:
        raise ValueError("wrong source")
    cache = Path(payload["cache_path"])
    if cache.stat().st_size != int(payload["byte_size"]):
        raise ValueError("cache byte size mismatch")
    if sys.argv[4] == "1":
        source_dataset = Path(sys.argv[3])
        digest = hashlib.sha256((source_dataset / "meta" / "topcam_rectification.json").read_bytes()).hexdigest()
        if payload.get("topcam_rectification", {}).get("sha256") != digest:
            raise ValueError("topcam provenance mismatch")
        sys.path.insert(0, str(Path(sys.argv[5]) / "scripts"))
        import build_dino_memfd_feature_cache as cache_shared

        validation = cache_shared.validate_dataset(
            source_dataset,
            sys.argv[2],
            (
                "observation.images.head_cam",
                "observation.images.left_arm_cam",
                "observation.images.right_arm_cam",
            ),
        )
        cached_dataset = payload.get("dataset", {})
        if cached_dataset.get("info_sha256") != validation.info_sha256:
            raise ValueError("cache info.json provenance mismatch")
        if cached_dataset.get("source_fingerprint") != validation.source_fingerprint:
            raise ValueError("cache source fingerprint mismatch")
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
source_ready || { printf 'Source conversion did not produce the required contract\n' >&2; exit 1; }

if ! v3_ready; then
  bash "${V3_BUILD_SCRIPT}"
fi
v3_ready || { printf 'V3 construction did not produce the required contract\n' >&2; exit 1; }

if ! cache_ready; then
  bash "${CACHE_BUILD_SCRIPT}"
fi
cache_ready || { printf 'Disk DINO cache did not produce the required contract\n' >&2; exit 1; }

start_if_stopped "${RAW_TRAIN_PROGRAM}"
start_if_stopped "${V3_TRAIN_PROGRAM}"
sleep 10
for program in "${RAW_TRAIN_PROGRAM}" "${V3_TRAIN_PROGRAM}"; do
  state="$(program_state "${program}")"
  if [[ "${state}" != RUNNING && "${state}" != STARTING ]]; then
    printf '%s failed to remain running after launch: %s\n' "${program}" "${state:-unknown}" >&2
    exit 1
  fi
done
printf '[%s] normal and V3 100k trainings both launched; durable cache retained\n' "$(date '+%F %T')"

while true; do
  raw_state="$(program_state "${RAW_TRAIN_PROGRAM}")"
  v3_state="$(program_state "${V3_TRAIN_PROGRAM}")"
  raw_active=false
  v3_active=false
  [[ "${raw_state}" == RUNNING || "${raw_state}" == STARTING ]] && raw_active=true
  [[ "${v3_state}" == RUNNING || "${v3_state}" == STARTING ]] && v3_active=true
  if [[ "${raw_active}" == true && "${v3_active}" == false && ! -f "${V3_SUCCESS}" ]]; then
    printf 'V3 training terminated without success; stopping raw peer to avoid an unmatched run.\n' >&2
    supervisorctl stop "${RAW_TRAIN_PROGRAM}" || true
    exit 1
  fi
  if [[ "${v3_active}" == true && "${raw_active}" == false && ! -f "${RAW_SUCCESS}" ]]; then
    printf 'Raw training terminated without success; stopping V3 peer to avoid an unmatched run.\n' >&2
    supervisorctl stop "${V3_TRAIN_PROGRAM}" || true
    exit 1
  fi
  if [[ "${raw_active}" == true || "${v3_active}" == true ]]; then
    printf '[%s] training states: raw=%s v3=%s\n' \
      "$(date '+%F %T')" "${raw_state:-unknown}" "${v3_state:-unknown}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  printf '[%s] terminal states: raw=%s v3=%s; disk cache deliberately retained\n' \
    "$(date '+%F %T')" "${raw_state:-unknown}" "${v3_state:-unknown}"
  if [[ "${raw_state}" == EXITED && "${v3_state}" == EXITED && -f "${RAW_SUCCESS}" && -f "${V3_SUCCESS}" ]]; then
    exit 0
  fi
  printf 'A training exited without its success sentinel. raw_success=%s v3_success=%s\n' \
    "$(test -f "${RAW_SUCCESS}" && printf yes || printf no)" \
    "$(test -f "${V3_SUCCESS}" && printf yes || printf no)" >&2
  exit 1
done
