#!/usr/bin/env bash
set -euo pipefail

# Keep the full frozen-DINO cache and its consumer in one supervised process
# group. The cache is anonymous RAM, so it must stay alive for the entire
# training run and must not be launched as an unrelated detached process.

if (( $# > 0 )); then
  printf 'Configure this script with environment variables, not positional arguments.\n' >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_decoder7_ddp128_100k}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
DINOV2_MODEL="${DINOV2_MODEL:-vit_base_patch16_dinov3.lvd1689m}"
DINOV2_PRETRAINED="${DINOV2_PRETRAINED:-true}"
CACHE_BUILDER="${CACHE_BUILDER:-${REPO_ROOT}/scripts/build_dino_memfd_feature_cache.py}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/dino_feature_cache}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${CACHE_DIR}/${RUN_ID}.json}"
CACHE_LOG="${CACHE_LOG:-${CACHE_DIR}/${RUN_ID}.log}"
DINO_CACHE_BATCH_SIZE="${DINO_CACHE_BATCH_SIZE:-32}"
DINO_CACHE_MIN_FREE_GIB="${DINO_CACHE_MIN_FREE_GIB:-64}"
DINO_CACHE_DEVICES="${DINO_CACHE_DEVICES:-0,1}"
# This affects only the one-off raw-video -> frozen-DINO cache conversion.
# ACT subsequently reads feature maps directly, so keeping TorchCodec CPU
# decoders alive here lowers cache-build latency without changing the inputs
# or the training-step semantics.
DINO_CACHE_VIDEO_BACKEND="${DINO_CACHE_VIDEO_BACKEND:-torchcodec}"
CACHE_POLL_SECONDS="${CACHE_POLL_SECONDS:-5}"
CACHE_WAIT_SECONDS="${CACHE_WAIT_SECONDS:-7200}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/scripts/train_robot8_20260721_act_dinov3_ddp128_all23_100k.sh}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CACHE_BUILDER}" ]]; then
  printf 'Missing DINO cache builder: %s\n' "${CACHE_BUILDER}" >&2
  exit 1
fi
if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  printf 'Missing fresh training script: %s\n' "${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  printf 'Missing all-23D dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to allocate a cache for an existing output: %s\n' "${OUTPUT_DIR}" >&2
  printf 'Use a new RUN_ID, or deliberately resume from a saved checkpoint.\n' >&2
  exit 1
fi
if ! [[ "${DINO_CACHE_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( DINO_CACHE_BATCH_SIZE < 1 )); then
  printf 'DINO_CACHE_BATCH_SIZE must be a positive integer.\n' >&2
  exit 1
fi
if ! [[ "${DINO_CACHE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'DINO_CACHE_DEVICES must be one or more comma-separated CUDA indexes, got %s.\n' \
    "${DINO_CACHE_DEVICES}" >&2
  exit 1
fi
case "${DINO_CACHE_VIDEO_BACKEND}" in
  pyav|torchcodec) ;;
  *)
    printf 'DINO_CACHE_VIDEO_BACKEND must be pyav or torchcodec, got %s.\n' \
      "${DINO_CACHE_VIDEO_BACKEND}" >&2
    exit 1
    ;;
esac
case "${DINOV2_PRETRAINED}" in
  true|false) ;;
  *)
    printf 'DINOV2_PRETRAINED must be true or false, got %s.\n' "${DINOV2_PRETRAINED}" >&2
    exit 1
    ;;
esac
if ! [[ "${CACHE_POLL_SECONDS}" =~ ^[0-9]+$ && "${CACHE_WAIT_SECONDS}" =~ ^[0-9]+$ ]] || \
  (( CACHE_POLL_SECONDS < 1 || CACHE_WAIT_SECONDS < CACHE_POLL_SECONDS )); then
  printf 'CACHE_POLL_SECONDS and CACHE_WAIT_SECONDS must be positive, with wait >= poll.\n' >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
# Vast base images export a global /workspace/.hf_home.  This run carries its
# verified DINO weights beside the repository, so do not silently inherit a
# different cache.  REPO_HF_HOME remains an explicit opt-in override.
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "${CACHE_DIR}"

# The cache builder replaces failed manifests itself, but its worker startup
# takes longer than the first poll below.  Leaving a prior `failed` manifest in
# place therefore makes this wrapper mistake the *previous* attempt for the
# newly launched cache process and terminate it before replacement occurs.
# There cannot be a reusable cache for this fresh output directory; archive a
# terminal manifest before launching, while refusing to disturb an active one.
if [[ -f "${CACHE_MANIFEST}" ]]; then
  prior_cache_status="$(
    "${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
    print(payload.get("status", "unknown"))
except Exception:
    print("unknown")
PY
  )"
  case "${prior_cache_status}" in
    failed|stopped|unknown|"")
      archived_manifest="${CACHE_MANIFEST}.stale.$(date '+%Y%m%d-%H%M%S').$$"
      mv -- "${CACHE_MANIFEST}" "${archived_manifest}"
      printf 'Archived prior terminal DINO cache manifest (%s): %s\n' \
        "${prior_cache_status:-unknown}" "${archived_manifest}" >&2
      ;;
    *)
      printf 'Refusing to replace non-terminal DINO cache manifest (status=%s): %s\n' \
        "${prior_cache_status}" "${CACHE_MANIFEST}" >&2
      exit 1
      ;;
  esac
fi

cache_cmd=(
  "${VENV_DIR}/bin/python"
  "${CACHE_BUILDER}"
  "--dataset-root=${DATASET_ROOT}"
  "--repo-id=${DATASET_REPO_ID}"
  "--model=${DINOV2_MODEL}"
  --cameras=head_cam,left_arm_cam,right_arm_cam
  "--manifest=${CACHE_MANIFEST}"
  "--devices=${DINO_CACHE_DEVICES}"
  "--batch-size=${DINO_CACHE_BATCH_SIZE}"
  "--video-backend=${DINO_CACHE_VIDEO_BACKEND}"
  "--min-free-gib=${DINO_CACHE_MIN_FREE_GIB}"
  --replace-stale-manifest
)
if [[ "${DINOV2_PRETRAINED}" == false ]]; then
  cache_cmd+=(--no-pretrained)
fi

printf '%q ' "${cache_cmd[@]}" >"${CACHE_LOG}.command"
printf '\n' >>"${CACHE_LOG}.command"
printf '[%s] building shared frozen-DINO cache: %s\n' "$(date '+%F %T')" "${CACHE_MANIFEST}" | tee "${CACHE_LOG}"
"${cache_cmd[@]}" >>"${CACHE_LOG}" 2>&1 &
cache_pid=$!

cleanup_cache() {
  if kill -0 "${cache_pid}" 2>/dev/null; then
    printf '[%s] stopping DINO cache server pid=%s\n' "$(date '+%F %T')" "${cache_pid}" | tee -a "${CACHE_LOG}"
    kill -TERM "${cache_pid}" 2>/dev/null || true
    wait "${cache_pid}" || true
  fi
}
on_signal() {
  exit 130
}
trap cleanup_cache EXIT
trap on_signal INT TERM

cache_ready=false
attempts=$(( CACHE_WAIT_SECONDS / CACHE_POLL_SECONDS ))
for ((attempt=1; attempt<=attempts; attempt++)); do
  if [[ -f "${CACHE_MANIFEST}" ]]; then
    read -r cache_status cache_ready < <(
      "${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("status", "unknown"), "true" if payload.get("ready") is True else "false")
PY
    )
    if [[ "${cache_ready}" == true ]]; then
      break
    fi
    if [[ "${cache_status}" == failed || "${cache_status}" == stopped ]]; then
      printf 'DINO cache server ended with status=%s. Recent log follows:\n' "${cache_status}" >&2
      tail -n 120 "${CACHE_LOG}" >&2 || true
      exit 1
    fi
  fi
  if ! kill -0 "${cache_pid}" 2>/dev/null; then
    printf 'DINO cache server exited before readiness. Recent log follows:\n' >&2
    tail -n 120 "${CACHE_LOG}" >&2 || true
    exit 1
  fi
  if (( attempt % 12 == 0 )); then
    printf '[%s] cache status=%s ready=%s; waiting...\n' \
      "$(date '+%F %T')" "${cache_status:-building}" "${cache_ready}" | tee -a "${CACHE_LOG}"
  fi
  sleep "${CACHE_POLL_SECONDS}"
done
if [[ "${cache_ready}" != true ]]; then
  printf 'Timed out waiting %ss for DINO feature cache. Recent log follows:\n' "${CACHE_WAIT_SECONDS}" >&2
  tail -n 120 "${CACHE_LOG}" >&2 || true
  exit 1
fi

printf '[%s] DINO cache is ready; starting fresh all-23D training.\n' "$(date '+%F %T')" | tee -a "${CACHE_LOG}"
export RUN_ID DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR VENV_DIR DINOV2_MODEL DINOV2_PRETRAINED
export DINO_FEATURE_CACHE_MANIFEST="${CACHE_MANIFEST}"
bash "${TRAIN_SCRIPT}"
