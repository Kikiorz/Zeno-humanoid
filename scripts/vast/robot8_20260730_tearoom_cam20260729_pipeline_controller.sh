#!/usr/bin/env bash
# Supervisor-owned 07-30 pipeline:
# 30 Hz rectified conversion -> V3 labels + shared frozen-DINO cache in
# parallel -> one normal and one V3 single-GPU training job.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260730_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
V3_REPO_ID="${V3_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
V3_DATASET="${V3_DATASET:-${LEROBOT_ROOT}/${V3_REPO_ID}}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/dino_feature_cache}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${CACHE_DIR}/robot8_20260730_tearoom_topcam_left_cam20260729_shared_dinov3_disk.json}"
CACHE_FILE="${CACHE_FILE:-${CACHE_DIR}/robot8_20260730_tearoom_topcam_left_cam20260729_shared_dinov3.f16}"
OLD_CACHE_FILE="${OLD_CACHE_FILE:-${CACHE_DIR}/robot8_20260729_tearoom_topcam_left_cam20260729_normal_dinov3.f16}"
OLD_CACHE_MANIFEST="${OLD_CACHE_MANIFEST:-${CACHE_DIR}/robot8_20260729_tearoom_topcam_left_cam20260729_normal_dinov3_disk.json}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260730_tearoom_cam20260729_convert.sh}"
V3_BUILD_SCRIPT="${V3_BUILD_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260730_tearoom_cam20260729_v3_build.sh}"
CACHE_BUILD_SCRIPT="${CACHE_BUILD_SCRIPT:-${REPO_ROOT}/scripts/vast/robot8_20260730_tearoom_cam20260729_shared_dino_disk_cache.sh}"
NORMAL_PROGRAM="${NORMAL_PROGRAM:-robot8_20260730_tearoom_cam20260729_normal_train}"
V3_PROGRAM="${V3_PROGRAM:-robot8_20260730_tearoom_cam20260729_v3_train}"
NORMAL_SUCCESS="${NORMAL_SUCCESS:-${REPO_ROOT}/outputs/train/robot8_20260730_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_b32_100k_seqcache/TRAINING_SUCCEEDED}"
V3_SUCCESS="${V3_SUCCESS:-${REPO_ROOT}/outputs/train/robot8_20260730_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k_seqcache/TRAINING_SUCCEEDED}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPECTED_CALIBRATION_SHA256="6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342"
MIN_SECONDARY_ACCEPTED_FRACTION="${MIN_SECONDARY_ACCEPTED_FRACTION:-0.70}"
MIN_SECONDARY_TV_REDUCTION="${MIN_SECONDARY_TV_REDUCTION:-0.20}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 1 )); then
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 1
fi

program_state() {
  local state
  state="$(supervisorctl status "$1" 2>/dev/null | awk '{print $2}' || true)"
  printf '%s' "${state}"
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
  "${VENV_DIR}/bin/python" - "${SOURCE_DATASET}" "${EXPECTED_CALIBRATION_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_calibration_sha = sys.argv[2]
try:
    info = json.loads((root / "meta" / "info.json").read_text())
    top = json.loads((root / "meta" / "topcam_rectification.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if info.get("fps") != 30 or info.get("total_episodes") != 4 or int(info.get("total_frames", 0)) < 50_000:
    raise SystemExit(1)
if top.get("profile") != "cam_20260729" or top.get("pipeline") != (
    "split_left_right_then_cam_20260729_opencv_fisheye_rectify_then_independent_left_right_rgb_resize"
):
    raise SystemExit(1)
if top.get("head_camera_feature_to_eye") != {"head_cam": "left"}:
    raise SystemExit(1)
if top.get("selected_model_topcam_eye") != "left" or top.get("head_stereo_model_resize_mode") != "letterbox":
    raise SystemExit(1)
if top.get("spatial_crop") != {"x": 20, "y": 0, "width": 1240, "height": 620}:
    raise SystemExit(1)
if top.get("rectified_size_per_eye") != {"width": 1280, "height": 720}:
    raise SystemExit(1)
if top.get("calibration_sha256") != expected_calibration_sha:
    raise SystemExit(1)
PY
}

v3_ready() {
  "${VENV_DIR}/bin/python" - "${V3_DATASET}" "${MIN_SECONDARY_ACCEPTED_FRACTION}" "${MIN_SECONDARY_TV_REDUCTION}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_fraction = float(sys.argv[2])
min_reduction = float(sys.argv[3])
try:
    info = json.loads((root / "meta" / "info.json").read_text())
    summary = json.loads((root / "meta" / "base_anchor_odom_v3_decoupled_smooth" / "generation_summary.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if info.get("fps") != 30 or summary.get("version") != "base_anchor_odom_v3_decoupled_smooth":
    raise SystemExit(1)
if float(summary.get("fps", -1)) != 30:
    raise SystemExit(1)
secondary = summary.get("secondary_decoupled_smoothing", {})
if float(secondary.get("accepted_action_frame_fraction", -1)) < min_fraction:
    raise SystemExit(1)
if float(secondary.get("velocity_total_variation_reduction_vs_v2", -1)) < min_reduction:
    raise SystemExit(1)
PY
  [[ $? -eq 0 ]] || return 1
  "${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_clean/verify_shared_visual_dataset.py" \
    --source-dataset "${SOURCE_DATASET}" \
    --derived-dataset "${V3_DATASET}"
}

cache_ready() {
  "${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" "${CACHE_FILE}" "${SOURCE_REPO_ID}" "${SOURCE_DATASET}" "${REPO_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
cache_path = Path(sys.argv[2])
repo_id = sys.argv[3]
source = Path(sys.argv[4])
repo_root = Path(sys.argv[5])
try:
    payload = json.loads(manifest_path.read_text())
    source_topcam = (source / "meta" / "topcam_rectification.json").read_bytes()
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if payload.get("storage") != "disk" or payload.get("ready") is not True or payload.get("status") != "ready":
    raise SystemExit(1)
if payload.get("dataset", {}).get("repo_id") != repo_id:
    raise SystemExit(1)
if cache_path.stat().st_size != int(payload.get("byte_size", -1)):
    raise SystemExit(1)
if Path(payload.get("cache_path", "")) != cache_path:
    raise SystemExit(1)
cached_topcam = payload.get("topcam_rectification", {}).get("sha256")
if cached_topcam != hashlib.sha256(source_topcam).hexdigest():
    raise SystemExit(1)
sys.path.insert(0, str(repo_root / "scripts"))
import build_dino_memfd_feature_cache as cache_shared

validation = cache_shared.validate_dataset(
    source,
    repo_id,
    (
        "observation.images.head_cam",
        "observation.images.left_arm_cam",
        "observation.images.right_arm_cam",
    ),
)
cached_dataset = payload.get("dataset", {})
if cached_dataset.get("info_sha256") != validation.info_sha256:
    raise SystemExit(1)
if cached_dataset.get("source_fingerprint") != validation.source_fingerprint:
    raise SystemExit(1)
index = payload.get("data_index", {})
if index.get("first") != 0 or index.get("count") != validation.total_frames:
    raise SystemExit(1)
PY
}

clear_stale_new_cache() {
  if [[ ! -e "${CACHE_FILE}" && ! -e "${CACHE_MANIFEST}" ]]; then
    return 0
  fi
  if ! "${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text())
except json.JSONDecodeError:
    raise SystemExit(0)
raise SystemExit(1 if payload.get("ready") is True else 0)
PY
  then
    printf 'Refusing to remove a ready-but-invalid new DINO cache; inspect %s\n' "${CACHE_MANIFEST}" >&2
    return 1
  fi
  if [[ -e "${CACHE_FILE}" ]] && command -v fuser >/dev/null 2>&1 && fuser -s "${CACHE_FILE}"; then
    printf 'Refusing to remove an in-use incomplete DINO cache: %s\n' "${CACHE_FILE}" >&2
    return 1
  fi
  printf '[%s] removing only the non-ready 07-30 cache artifacts left by an interrupted build\n' \
    "$(date '+%F %T')"
  rm -f -- "${CACHE_FILE}" "${CACHE_MANIFEST}"
}

remove_rebuildable_old_cache() {
  if [[ ! -e "${OLD_CACHE_FILE}" && ! -e "${OLD_CACHE_MANIFEST}" ]]; then
    return 0
  fi
  if command -v fuser >/dev/null 2>&1 && fuser -s "${OLD_CACHE_FILE}"; then
    printf 'Refusing to delete an in-use old DINO cache: %s\n' "${OLD_CACHE_FILE}" >&2
    return 1
  fi
  printf '[%s] removing the unreferenced, rebuildable 2026-07-29 DINO cache to free disk for 07-30\n' \
    "$(date '+%F %T')"
  rm -f -- "${OLD_CACHE_FILE}" "${OLD_CACHE_MANIFEST}"
}

if ! source_ready; then
  (
    export DATASET_REPO_ID="${SOURCE_REPO_ID}" DATASET_ROOT="${SOURCE_DATASET}"
    bash "${CONVERT_SCRIPT}"
  )
fi
source_ready || { printf 'Source conversion did not produce the required 30 Hz camera contract\n' >&2; exit 1; }

# V3 modifies labels only, so it can run on CPU in parallel with the one-time
# two-GPU cache build.  Both must pass before either training service starts.
v3_pid=''
cache_pid=''
if ! v3_ready; then
  (
    export SOURCE_REPO_ID SOURCE_DATASET
    export DATASET_REPO_ID="${V3_REPO_ID}" DATASET_ROOT="${V3_DATASET}"
    export MIN_SECONDARY_ACCEPTED_FRACTION MIN_SECONDARY_TV_REDUCTION
    bash "${V3_BUILD_SCRIPT}"
  ) &
  v3_pid="$!"
fi
if ! cache_ready; then
  clear_stale_new_cache
  remove_rebuildable_old_cache
  (
    export SOURCE_REPO_ID SOURCE_DATASET CACHE_DIR CACHE_MANIFEST CACHE_FILE
    bash "${CACHE_BUILD_SCRIPT}"
  ) &
  cache_pid="$!"
fi

stage_status=0
if [[ -n "${v3_pid}" ]] && ! wait "${v3_pid}"; then
  printf 'V3 construction failed; cache result will be retained only for inspection.\n' >&2
  stage_status=1
fi
if [[ -n "${cache_pid}" ]] && ! wait "${cache_pid}"; then
  printf 'DINO cache construction failed.\n' >&2
  stage_status=1
fi
(( stage_status == 0 )) || exit "${stage_status}"
v3_ready || { printf 'V3 construction did not pass shared-visual validation\n' >&2; exit 1; }
cache_ready || { printf 'DINO cache did not pass source-provenance validation\n' >&2; exit 1; }

start_if_stopped "${NORMAL_PROGRAM}"
start_if_stopped "${V3_PROGRAM}"
sleep 10
for program in "${NORMAL_PROGRAM}" "${V3_PROGRAM}"; do
  state="$(program_state "${program}")"
  if [[ "${state}" != RUNNING && "${state}" != STARTING ]]; then
    printf '%s failed to remain running after launch: %s\n' "${program}" "${state:-unknown}" >&2
    exit 1
  fi
done
printf '[%s] normal GPU0 and V3 GPU1 training both launched\n' "$(date '+%F %T')"

while true; do
  normal_state="$(program_state "${NORMAL_PROGRAM}")"
  v3_state="$(program_state "${V3_PROGRAM}")"
  normal_active=false
  v3_active=false
  [[ "${normal_state}" == RUNNING || "${normal_state}" == STARTING ]] && normal_active=true
  [[ "${v3_state}" == RUNNING || "${v3_state}" == STARTING ]] && v3_active=true
  if [[ "${normal_active}" == true || "${v3_active}" == true ]]; then
    printf '[%s] training states: normal=%s v3=%s\n' \
      "$(date '+%F %T')" "${normal_state:-unknown}" "${v3_state:-unknown}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  printf '[%s] terminal states: normal=%s v3=%s\n' \
    "$(date '+%F %T')" "${normal_state:-unknown}" "${v3_state:-unknown}"
  if [[ "${normal_state}" == EXITED && "${v3_state}" == EXITED && -f "${NORMAL_SUCCESS}" && -f "${V3_SUCCESS}" ]]; then
    exit 0
  fi
  printf 'A peer terminated without a success sentinel. normal_success=%s v3_success=%s\n' \
    "$(test -f "${NORMAL_SUCCESS}" && printf yes || printf no)" \
    "$(test -f "${V3_SUCCESS}" && printf yes || printf no)" >&2
  exit 1
done
