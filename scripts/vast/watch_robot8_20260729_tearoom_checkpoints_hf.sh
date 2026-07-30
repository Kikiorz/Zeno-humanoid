#!/usr/bin/env bash
# Continuously publish and fetch only fully committed TeaRoom checkpoints.
#
# The remote trainer updates checkpoints/last only after a checkpoint is
# complete.  This watcher uses that marker, validates the numbered checkpoint,
# uploads it with the existing HF synchronization helper on the remote host,
# then downloads only the deployment payload to the exact local step path.
#
# Hugging Face credentials are never persisted or printed here.  A token from
# the local Hugging Face credential store is passed to the remote upload over
# SSH standard input only.  Set ONCE=true for a single safe polling pass.
set -euo pipefail

if (( $# > 0 )); then
  printf 'Configure this watcher with environment variables, not positional arguments.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_ID="${RUN_ID:-robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp32_b16_seqcache_60k}"

REMOTE_HOST="${REMOTE_HOST:-root@171.101.230.38}"
REMOTE_PORT="${REMOTE_PORT:-58289}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/workspace/2027icra}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-${REMOTE_REPO_ROOT}/outputs/train/${RUN_ID}}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/venv/main/bin/python}"
REMOTE_SYNC_SCRIPT="${REMOTE_SYNC_SCRIPT:-${REMOTE_REPO_ROOT}/scripts/sync_robot8_20260729_tearoom_checkpoint_hf.py}"

LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-${LOCAL_REPO_ROOT}}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
LOCAL_SYNC_SCRIPT="${LOCAL_SYNC_SCRIPT:-${LOCAL_REPO_ROOT}/scripts/sync_robot8_20260729_tearoom_checkpoint_hf.py}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
# Ignore historical checkpoints below this absolute step.  Set this to the
# next desired save point when taking over an already-running job.
MIN_STEP="${MIN_STEP:-0}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-15}"
ONCE="${ONCE:-false}"
# A comma-separated list such as 016000.  This is useful while a checkpoint is
# being manually published; once its local deployment files are complete it is
# skipped automatically even after the list is cleared.
SKIP_STEPS="${SKIP_STEPS:-}"
LOCK_FILE="${LOCK_FILE:-${LOCAL_ARTIFACT_ROOT}/outputs/train/${RUN_ID}/.checkpoint_hf_watcher.lock}"

MODEL_FILES=(
  model.safetensors
  config.json
  train_config.json
  policy_preprocessor.json
  policy_preprocessor_step_3_normalizer_processor.safetensors
  policy_postprocessor.json
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
)

die() {
  printf '%s\n' "$*" >&2
  exit 2
}

positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}

remote_quote() {
  local escaped="$1"
  escaped=${escaped//\'/\'\"\'\"\'}
  printf "'%s'" "${escaped}"
}

if ! positive_integer "${REMOTE_PORT}" || ! positive_integer "${POLL_SECONDS}" || \
   ! positive_integer "${CHECKPOINT_INTERVAL}" || ! positive_integer "${SSH_CONNECT_TIMEOUT}"; then
  die 'REMOTE_PORT, POLL_SECONDS, CHECKPOINT_INTERVAL, and SSH_CONNECT_TIMEOUT must be positive integers.'
fi
[[ "${MIN_STEP}" =~ ^[0-9]+$ ]] || die 'MIN_STEP must be a non-negative integer.'
if (( 10#${MIN_STEP} % 10#${CHECKPOINT_INTERVAL} != 0 )); then
  die 'MIN_STEP must be divisible by CHECKPOINT_INTERVAL.'
fi
case "${ONCE}" in
  true|false) ;;
  *) die 'ONCE must be true or false.' ;;
esac
[[ "${RUN_ID}" != */* && "${RUN_ID}" != . && "${RUN_ID}" != .. && -n "${RUN_ID}" ]] || \
  die 'RUN_ID must be one non-empty directory name.'
[[ "${REMOTE_REPO_ROOT}" == /* && "${REMOTE_RUN_DIR}" == /* ]] || die 'Remote paths must be absolute.'
[[ -f "${LOCAL_SYNC_SCRIPT}" ]] || die "Local synchronization helper is missing: ${LOCAL_SYNC_SCRIPT}"
command -v ssh >/dev/null || die 'ssh is required.'
command -v flock >/dev/null || die 'flock is required to prevent duplicate watchers.'
"${LOCAL_PYTHON}" -c 'import huggingface_hub' >/dev/null 2>&1 || \
  die "${LOCAL_PYTHON} cannot import huggingface_hub. Set LOCAL_PYTHON to the authenticated environment."

declare -A skipped_steps=()
if [[ -n "${SKIP_STEPS}" ]]; then
  IFS=',' read -r -a skip_values <<<"${SKIP_STEPS}"
  for skip_step in "${skip_values[@]}"; do
    [[ "${skip_step}" =~ ^[0-9]{6}$ ]] || die 'SKIP_STEPS must be a comma-separated list of six-digit steps.'
    skipped_steps["${skip_step}"]=1
  done
fi

mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  die "Another checkpoint watcher already holds ${LOCK_FILE}."
fi

ssh_cmd=(
  ssh
  -p "${REMOTE_PORT}"
  -o BatchMode=yes
  -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
  "${REMOTE_HOST}"
)

local_checkpoint_dir() {
  local step="$1"
  printf '%s/outputs/train/%s/checkpoints/%s' \
    "${LOCAL_ARTIFACT_ROOT}" "${RUN_ID}" "${step}"
}

local_deploy_complete() {
  local step="$1"
  local checkpoint pretrained file
  checkpoint="$(local_checkpoint_dir "${step}")"
  pretrained="${checkpoint}/pretrained_model"
  [[ "${checkpoint}" == "${LOCAL_ARTIFACT_ROOT}/outputs/train/${RUN_ID}/checkpoints/${step}" ]] || return 1
  [[ "$(basename "${checkpoint}")" == "${step}" ]] || return 1
  [[ ! -L "${checkpoint}" && ! -L "${pretrained}" ]] || return 1
  for file in "${MODEL_FILES[@]}"; do
    [[ -s "${pretrained}/${file}" ]] || return 1
  done
  return 0
}

local_deploy_has_files() {
  local step="$1"
  local pretrained file
  pretrained="$(local_checkpoint_dir "${step}")/pretrained_model"
  for file in "${MODEL_FILES[@]}"; do
    [[ -e "${pretrained}/${file}" ]] && return 0
  done
  return 1
}

get_hf_token() {
  "${LOCAL_PYTHON}" - <<'PY'
from huggingface_hub import get_token

token = get_token()
if not token:
    raise SystemExit(
        "No local Hugging Face token is available. Run `huggingface-cli login` in the local environment."
    )
print(token)
PY
}

remote_complete_steps() {
  local remote_command
  remote_command="bash -s -- $(remote_quote "${REMOTE_PYTHON}") $(remote_quote "${REMOTE_RUN_DIR}") $(remote_quote "${CHECKPOINT_INTERVAL}") $(remote_quote "${MIN_STEP}")"
  "${ssh_cmd[@]}" "${remote_command}" <<'REMOTE_CHECK'
set -euo pipefail
remote_python="$1"
run_dir="$2"
interval="$3"
min_step="$4"

"${remote_python}" - "${run_dir}" "${interval}" "${min_step}" <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
interval = int(sys.argv[2])
min_step = int(sys.argv[3])
checkpoints = run_dir / "checkpoints"
required_model_files = (
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
required_state_files = (
    "training_step.json",
    "optimizer_state.safetensors",
    "optimizer_param_groups.json",
    "rng_state.safetensors",
)

try:
    last_target = os.readlink(checkpoints / "last")
except OSError:
    raise SystemExit(0)
last_name = Path(last_target).name
if not re.fullmatch(r"[0-9]{6}", last_name):
    raise SystemExit(f"unsafe checkpoints/last target: {last_target!r}")
last_step = int(last_name)

try:
    candidates = sorted(checkpoints.iterdir(), key=lambda path: path.name)
except OSError:
    raise SystemExit(0)

for checkpoint in candidates:
    if not checkpoint.is_dir() or not re.fullmatch(r"[0-9]{6}", checkpoint.name):
        continue
    step = int(checkpoint.name)
    if step <= 0 or step < min_step or step > last_step or step % interval:
        continue
    pretrained = checkpoint / "pretrained_model"
    training_state = checkpoint / "training_state"
    try:
        required = [pretrained / name for name in required_model_files]
        required += [training_state / name for name in required_state_files]
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            continue
        payload = json.loads((training_state / "training_step.json").read_text(encoding="utf-8"))
        if int(payload.get("step", -1)) != step:
            continue
    except (OSError, ValueError, json.JSONDecodeError):
        continue
    print(f"{step:06d}")
PY
REMOTE_CHECK
}

upload_remote_step() {
  local step="$1"
  local token remote_program remote_command
  token="$(get_hf_token)" || return 1
  [[ -n "${token}" ]] || { printf 'Refusing remote upload without a Hugging Face token.\n' >&2; return 1; }

  remote_program='set -euo pipefail
IFS= read -r hf_token
[[ -n "${hf_token}" ]] || { printf "Missing Hugging Face token on standard input.\\n" >&2; exit 1; }
export HF_TOKEN="${hf_token}"
exec "$1" "$2" upload --step "$3" --repo-root "$4" --artifact-root "$4" --run-id "$5"'
  remote_command="bash -c $(remote_quote "${remote_program}") $(remote_quote checkpoint-hf-upload) $(remote_quote "${REMOTE_PYTHON}") $(remote_quote "${REMOTE_SYNC_SCRIPT}") $(remote_quote "${step}") $(remote_quote "${REMOTE_REPO_ROOT}") $(remote_quote "${RUN_ID}")"

  if ! printf '%s\n' "${token}" | "${ssh_cmd[@]}" "${remote_command}"; then
    token=''
    unset token
    return 1
  fi
  token=''
  unset token
}

download_local_step() {
  local step="$1"
  local token
  local command=(
    "${LOCAL_PYTHON}" "${LOCAL_SYNC_SCRIPT}" download
    --step "${step}"
    --run-id "${RUN_ID}"
    --repo-root "${LOCAL_REPO_ROOT}"
    --artifact-root "${LOCAL_ARTIFACT_ROOT}"
    --deploy-only
  )
  if local_deploy_has_files "${step}"; then
    command+=(--overwrite)
  fi
  token="$(get_hf_token)" || return 1
  [[ -n "${token}" ]] || { printf 'Refusing local download without a Hugging Face token.\n' >&2; return 1; }
  if ! HF_TOKEN="${token}" "${command[@]}"; then
    token=''
    unset token
    return 1
  fi
  token=''
  unset token
  local_deploy_complete "${step}"
}

sync_ready_steps() {
  local steps step
  if ! steps="$(remote_complete_steps)"; then
    printf '[%s] remote checkpoint probe failed; it will be retried.\n' "$(date '+%F %T')" >&2
    return 1
  fi
  if [[ -z "${steps}" ]]; then
    printf '[%s] no complete new %s-step checkpoint published through checkpoints/last.\n' \
      "$(date '+%F %T')" "${CHECKPOINT_INTERVAL}"
    return 0
  fi
  while IFS= read -r step; do
    [[ "${step}" =~ ^[0-9]{6}$ ]] || continue
    if local_deploy_complete "${step}"; then
      printf '[%s] checkpoint %s is already locally deployable; skipping.\n' \
        "$(date '+%F %T')" "${step}"
      continue
    fi
    if [[ -n "${skipped_steps[${step}]+x}" ]]; then
      printf '[%s] checkpoint %s is in SKIP_STEPS; leaving its manual publication untouched.\n' \
        "$(date '+%F %T')" "${step}"
      continue
    fi
    printf '[%s] checkpoint %s is complete remotely; publishing to Hugging Face.\n' \
      "$(date '+%F %T')" "${step}"
    if ! upload_remote_step "${step}"; then
      printf '[%s] remote Hugging Face upload failed for checkpoint %s; it will be retried.\n' \
        "$(date '+%F %T')" "${step}" >&2
      return 1
    fi
    printf '[%s] downloading deployable checkpoint %s into its strict local path.\n' \
      "$(date '+%F %T')" "${step}"
    if ! download_local_step "${step}"; then
      printf '[%s] local checkpoint %s download/validation failed; it will be retried.\n' \
        "$(date '+%F %T')" "${step}" >&2
      return 1
    fi
    printf '[%s] checkpoint %s verified: model and deployment files are complete locally.\n' \
      "$(date '+%F %T')" "${step}"
  done <<<"${steps}"
}

printf '[%s] watching %s on %s:%s every %ss from step %s; local deploy root: %s\n' \
  "$(date '+%F %T')" "${RUN_ID}" "${REMOTE_HOST}" "${REMOTE_PORT}" "${POLL_SECONDS}" "${MIN_STEP}" \
  "${LOCAL_ARTIFACT_ROOT}/outputs/train/${RUN_ID}"

while true; do
  cycle_status=0
  sync_ready_steps || cycle_status=$?
  if [[ "${ONCE}" == true ]]; then
    exit "${cycle_status}"
  fi
  sleep "${POLL_SECONDS}"
done
