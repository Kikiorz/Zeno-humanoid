#!/usr/bin/env bash
set -euo pipefail

# Continue the normal (non-base-weighted) DINOv3 ACT run from a full training
# checkpoint.  `STEPS` is a global optimizer-step target, not additional steps.
# On the 24 GiB RTX 4090 D, a real one-step resume probe OOMed at batch 8;
# batch 4 completed, so it is the safe default.

if (( $# > 0 )); then
  printf 'Configure this script with environment variables (for example RESUME_STEP=020000), not positional arguments.\n' >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot-qrp312}"

RUN_ID="${RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_frozen_lift_waist_decoder7_10k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
RESUME_STEP="${RESUME_STEP:-010000}"
TOTAL_STEPS="${TOTAL_STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${RESUME_STEP}"
CONFIG_PATH="${CHECKPOINT_DIR}/pretrained_model/train_config.json"
TRAINING_STATE_DIR="${CHECKPOINT_DIR}/training_state"
LOG_DIR="${REPO_ROOT}/scripts/train_log/${RUN_ID}_resume_to_${TOTAL_STEPS}_b${BATCH_SIZE}"
LOG_PATH="${LOG_DIR}/train.log"

if [[ ! -f "${CONFIG_PATH}" || ! -d "${TRAINING_STATE_DIR}" ]]; then
  printf 'Missing full resume checkpoint: %s\n' "${CHECKPOINT_DIR}" >&2
  exit 1
fi
if ! [[ "${RESUME_STEP}" =~ ^[0-9]+$ ]] || ! [[ "${TOTAL_STEPS}" =~ ^[0-9]+$ ]]; then
  printf 'RESUME_STEP and TOTAL_STEPS must be zero-padded/plain decimal integers.\n' >&2
  exit 1
fi
if (( 10#${RESUME_STEP} >= TOTAL_STEPS )); then
  printf 'RESUME_STEP (%s) must be below TOTAL_STEPS (%s).\n' "${RESUME_STEP}" "${TOTAL_STEPS}" >&2
  exit 1
fi
if (( BATCH_SIZE < 1 )); then
  printf 'BATCH_SIZE must be positive.\n' >&2
  exit 1
fi
if (( SAVE_FREQ < 1 || TOTAL_STEPS % SAVE_FREQ != 0 )); then
  printf 'SAVE_FREQ must divide TOTAL_STEPS so requested checkpoints are exact.\n' >&2
  exit 1
fi

# Avoid silently overwriting a checkpoint created by a prior interrupted run.
for ((step=SAVE_FREQ; step<=TOTAL_STEPS; step+=SAVE_FREQ)); do
  step_name="$(printf '%06d' "${step}")"
  if (( step > 10#${RESUME_STEP} )) && [[ -e "${OUTPUT_DIR}/checkpoints/${step_name}" ]]; then
    printf 'Refusing to overwrite existing future checkpoint: %s\n' \
      "${OUTPUT_DIR}/checkpoints/${step_name}" >&2
    printf 'Set RESUME_STEP to that completed checkpoint to continue safely.\n' >&2
    exit 1
  fi
done

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy FTP_PROXY ftp_proxy

mkdir -p "${LOG_DIR}"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python3 -m lerobot.scripts.lerobot_train
  --config_path="${CONFIG_PATH}"
  --resume=true
  --steps="${TOTAL_STEPS}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --prefetch_factor="${PREFETCH_FACTOR}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --eval_freq=0
  --wandb.enable=false
)

printf '%q ' "${cmd[@]}" >"${LOG_DIR}/command.txt"
printf '\n' >>"${LOG_DIR}/command.txt"
printf '[%s] resume=%s target=%s batch=%s save_freq=%s\n' \
  "$(date '+%F %T')" "${RESUME_STEP}" "${TOTAL_STEPS}" "${BATCH_SIZE}" "${SAVE_FREQ}" \
  | tee -a "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
