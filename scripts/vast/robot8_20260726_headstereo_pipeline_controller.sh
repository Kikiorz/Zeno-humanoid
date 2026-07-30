#!/usr/bin/env bash
# Supervisor-managed sequencer for the corrected-data dual-GPU pipeline.
# It makes no model/data modifications itself; each irreversible stage is a
# separate fail-closed supervisor program with a fresh output path.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_4cam_640x480_headstereo_rectified_crop_lr_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/Data/lerobot/${SOURCE_REPO_ID}}"
V3_REPO_ID="${V3_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
V3_DATASET="${V3_DATASET:-${REPO_ROOT}/Data/lerobot/${V3_REPO_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260726_headstereo_rectified_lr_shared_dinov3.json}"
POLL_SECONDS="${POLL_SECONDS:-30}"

CONVERT_PROGRAM="robot8_20260726_headstereo_rectified_convert"
V3_BUILD_PROGRAM="robot8_20260726_headstereo_rectified_v3_build"
CACHE_PROGRAM="robot8_20260726_headstereo_shared_dino_cache"
RAW_TRAIN_PROGRAM="robot8_20260726_headstereo_rectified_raw_train"
V3_TRAIN_PROGRAM="robot8_20260726_headstereo_rectified_v3_train"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
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
    EXITED|FATAL|BACKOFF|UNKNOWN|"")
      printf 'Cannot safely start %s because supervisor state is %s\n' "${program}" "${state:-unknown}" >&2
      return 1
      ;;
    *)
      printf 'Unexpected supervisor state for %s: %s\n' "${program}" "${state}" >&2
      return 1
      ;;
  esac
}

raw_contract_ready() {
  "${VENV_DIR}/bin/python" - "${SOURCE_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
try:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    topcam = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if {key: info.get(key) for key in ("fps", "total_episodes", "total_frames")} != {
    "fps": 20, "total_episodes": 20, "total_frames": 44_490
}:
    raise SystemExit(1)
if topcam.get("output_eyes") != ["left", "right"]:
    raise SystemExit(1)
if topcam.get("head_camera_feature_to_eye") != {"head_cam": "left", "head_cam_right": "right"}:
    raise SystemExit(1)
if topcam.get("head_stereo_model_resize_mode") != "letterbox":
    raise SystemExit(1)
if topcam.get("generic_center_crop_applied_to_head_stereo") is not False:
    raise SystemExit(1)
PY
}

v3_contract_ready() {
  "${VENV_DIR}/bin/python" - "${V3_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
try:
    summary = json.loads((root / "meta" / "base_anchor_odom_v3_decoupled_smooth" / "generation_summary.json").read_text(encoding="utf-8"))
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
}

cache_ready() {
  "${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if payload.get("ready") is not True or payload.get("status") != "ready":
        raise ValueError("not ready")
    os.kill(int(payload["parent_pid"]), 0)
    if Path(payload["cache_path"]).stat().st_size != int(payload["byte_size"]):
        raise ValueError("cache byte size mismatch")
except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
    raise SystemExit(1)
PY
}

wait_for_contract() {
  local program="$1"
  local checker="$2"
  local stage="$3"
  while true; do
    if "${checker}"; then
      printf '[%s] %s contract is complete\n' "$(date '+%F %T')" "${stage}"
      return 0
    fi
    local state
    state="$(program_state "${program}")"
    case "${state}" in
      RUNNING|STARTING)
        printf '[%s] waiting for %s (%s)\n' "$(date '+%F %T')" "${stage}" "${state}"
        sleep "${POLL_SECONDS}"
        ;;
      STOPPED|EXITED|FATAL|BACKOFF|UNKNOWN|"")
        printf '%s did not produce its required contract; supervisor state=%s\n' \
          "${stage}" "${state:-unknown}" >&2
        return 1
        ;;
      *)
        printf 'Unexpected supervisor state while waiting for %s: %s\n' "${stage}" "${state}" >&2
        return 1
        ;;
    esac
  done
}

if ! raw_contract_ready; then
  start_if_stopped "${CONVERT_PROGRAM}"
  wait_for_contract "${CONVERT_PROGRAM}" raw_contract_ready "corrected raw conversion"
else
  printf '[%s] corrected raw dataset already passes contract\n' "$(date '+%F %T')"
fi

if ! v3_contract_ready; then
  start_if_stopped "${V3_BUILD_PROGRAM}"
  wait_for_contract "${V3_BUILD_PROGRAM}" v3_contract_ready "corrected V3 build"
else
  printf '[%s] corrected V3 dataset already passes contract\n' "$(date '+%F %T')"
fi

if ! cache_ready; then
  start_if_stopped "${CACHE_PROGRAM}"
  wait_for_contract "${CACHE_PROGRAM}" cache_ready "shared DINO cache"
else
  printf '[%s] shared DINO cache already passes contract\n' "$(date '+%F %T')"
fi

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
printf '[%s] corrected raw and V3 trainings are both launched; cache remains supervised separately\n' \
  "$(date '+%F %T')"

# Keep this controller alive as the cache lifecycle owner.  A memfd disappears
# only when its parent exits, so explicitly free its ~305 GiB once neither
# consumer is running; otherwise an already-finished experiment would reserve
# RAM indefinitely on the remote instance.
while true; do
  raw_state="$(program_state "${RAW_TRAIN_PROGRAM}")"
  v3_state="$(program_state "${V3_TRAIN_PROGRAM}")"
  raw_active=false
  v3_active=false
  [[ "${raw_state}" == RUNNING || "${raw_state}" == STARTING ]] && raw_active=true
  [[ "${v3_state}" == RUNNING || "${v3_state}" == STARTING ]] && v3_active=true
  if [[ "${raw_active}" == true || "${v3_active}" == true ]]; then
    printf '[%s] training states: raw=%s v3=%s; retaining shared cache\n' \
      "$(date '+%F %T')" "${raw_state:-unknown}" "${v3_state:-unknown}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  printf '[%s] training reached terminal states: raw=%s v3=%s; stopping shared cache\n' \
    "$(date '+%F %T')" "${raw_state:-unknown}" "${v3_state:-unknown}"
  cache_state="$(program_state "${CACHE_PROGRAM}")"
  if [[ "${cache_state}" == RUNNING || "${cache_state}" == STARTING ]]; then
    supervisorctl stop "${CACHE_PROGRAM}" || true
  fi
  if [[ "${raw_state}" == EXITED && "${v3_state}" == EXITED ]]; then
    exit 0
  fi
  exit 1
done
