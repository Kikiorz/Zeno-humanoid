#!/usr/bin/env bash
set -euo pipefail

# Resume the 3-camera frozen-DINOv3 ACT model with DDP.
# `BATCH_SIZE` is per GPU. With the default two ranks, 32 x 2 = global batch 64.

if (( $# > 0 )); then
  printf 'Configure this script with environment variables, not positional arguments.\n' >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_frozen_lift_waist_decoder7_10k}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_frozen_lift_waist}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
RESUME_STEP="${RESUME_STEP:-060000}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"

CONFIG_PATH="${OUTPUT_DIR}/checkpoints/${RESUME_STEP}/pretrained_model/train_config.json"
TRAINING_STATE_DIR="${OUTPUT_DIR}/checkpoints/${RESUME_STEP}/training_state"
LOG_DIR="${REPO_ROOT}/scripts/train_log/${RUN_ID}_ddp${NUM_PROCESSES}_global${GLOBAL_BATCH_SIZE}_resume_to_${TOTAL_STEPS}"
LOG_PATH="${LOG_DIR}/train.log"

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/accelerate" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" || ! -d "${TRAINING_STATE_DIR}" ]]; then
  printf 'Missing full resume checkpoint: %s\n' "${OUTPUT_DIR}/checkpoints/${RESUME_STEP}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  printf 'Missing dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi
if ! [[ "${RESUME_STEP}" =~ ^[0-9]+$ ]] || ! [[ "${TOTAL_STEPS}" =~ ^[0-9]+$ ]]; then
  printf 'RESUME_STEP and TOTAL_STEPS must be decimal integers.\n' >&2
  exit 1
fi
if (( 10#${RESUME_STEP} >= TOTAL_STEPS )); then
  printf 'RESUME_STEP (%s) must be below TOTAL_STEPS (%s).\n' "${RESUME_STEP}" "${TOTAL_STEPS}" >&2
  exit 1
fi
if (( BATCH_SIZE * NUM_PROCESSES != GLOBAL_BATCH_SIZE )); then
  printf 'Per-GPU batch %s x ranks %s is not global batch %s.\n' \
    "${BATCH_SIZE}" "${NUM_PROCESSES}" "${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi
if (( SAVE_FREQ < 1 || TOTAL_STEPS % SAVE_FREQ != 0 )); then
  printf 'SAVE_FREQ must divide TOTAL_STEPS.\n' >&2
  exit 1
fi

for ((step=SAVE_FREQ; step<=TOTAL_STEPS; step+=SAVE_FREQ)); do
  step_name="$(printf '%06d' "${step}")"
  if (( step > 10#${RESUME_STEP} )) && [[ -e "${OUTPUT_DIR}/checkpoints/${step_name}" ]]; then
    printf 'Refusing to overwrite future checkpoint: %s\n' \
      "${OUTPUT_DIR}/checkpoints/${step_name}" >&2
    exit 1
  fi
done

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "${LOG_DIR}"

cmd=(
  "${VENV_DIR}/bin/accelerate" launch
  --multi_gpu
  "--num_processes=${NUM_PROCESSES}"
  --mixed_precision=fp16
  -m lerobot.scripts.lerobot_train
  "--config_path=${CONFIG_PATH}"
  --resume=true
  "--dataset.root=${DATASET_ROOT}"
  "--output_dir=${OUTPUT_DIR}"
  "--steps=${TOTAL_STEPS}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--prefetch_factor=${PREFETCH_FACTOR}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  --eval_freq=0
  --wandb.enable=false
)

printf '%q ' "${cmd[@]}" >"${LOG_DIR}/command.txt"
printf '\n' >>"${LOG_DIR}/command.txt"
printf '[%s] resume=%s target=%s ranks=%s per_gpu_batch=%s global_batch=%s save_freq=%s\n' \
  "$(date '+%F %T')" "${RESUME_STEP}" "${TOTAL_STEPS}" "${NUM_PROCESSES}" \
  "${BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${SAVE_FREQ}" | tee -a "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
