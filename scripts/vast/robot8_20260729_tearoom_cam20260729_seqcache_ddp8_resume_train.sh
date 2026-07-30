#!/usr/bin/env bash
# Resume the immutable TeaRoom 002000 checkpoint on two GPUs with batch 8 per
# GPU (global batch 16). It deliberately preserves the frozen-DINO sequential
# cache sampler and the checkpoint's model/optimizer/RNG state.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
RUN_ID="${RUN_ID:-robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp32_b16_seqcache_60k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
RESUME_STEP="${RESUME_STEP:-2000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints/$(printf '%06d' "${RESUME_STEP}")}"
CONFIG_PATH="${CONFIG_PATH:-${CHECKPOINT_DIR}/pretrained_model/train_config.json}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260729_tearoom_topcam_left_cam20260729_normal_dinov3_disk.json}"
TOTAL_STEPS="${TOTAL_STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CACHE_LOCALITY_BATCH_SIZE="${CACHE_LOCALITY_BATCH_SIZE:-2048}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${REPO_ROOT}/scripts/train_log}"
LOG_RUN_ID="${RUN_ID}_resume_ddp16_b8"
LOG_DIR="${TRAIN_LOG_ROOT}/${LOG_RUN_ID}"
LOG_PATH="${LOG_DIR}/train.log"

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/accelerate" ]]; then
  printf 'Missing LeRobot environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" || ! -f "${CACHE_MANIFEST}" ]]; then
  printf 'Missing resume config or DINO cache manifest. config=%s cache=%s\n' "${CONFIG_PATH}" "${CACHE_MANIFEST}" >&2
  exit 1
fi
for file in \
  "${CHECKPOINT_DIR}/pretrained_model/model.safetensors" \
  "${CHECKPOINT_DIR}/pretrained_model/policy_preprocessor.json" \
  "${CHECKPOINT_DIR}/training_state/training_step.json" \
  "${CHECKPOINT_DIR}/training_state/optimizer_state.safetensors" \
  "${CHECKPOINT_DIR}/training_state/rng_state.safetensors"; do
  if [[ ! -f "${file}" ]]; then
    printf 'Incomplete resume checkpoint: %s\n' "${file}" >&2
    exit 1
  fi
done
if ! "${VENV_DIR}/bin/python" - "${CHECKPOINT_DIR}/training_state/training_step.json" "${RESUME_STEP}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(payload.get("step", -1)) != int(sys.argv[2]):
    raise SystemExit(f"resume step mismatch: {payload}")
PY
then
  exit 1
fi
if (( TOTAL_STEPS < RESUME_STEP || SAVE_FREQ < 1 || TOTAL_STEPS % SAVE_FREQ != 0 || BATCH_SIZE != 8 || CACHE_LOCALITY_BATCH_SIZE % (BATCH_SIZE * 2) != 0 )); then
  printf 'Expected two GPUs with per-GPU batch 8; invalid resume/train parameters.\n' >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"
printf '%s\n' \
  "RESUME_CHECKPOINT=${CHECKPOINT_DIR}" \
  "RESUME_STEP=${RESUME_STEP}" \
  "TOTAL_STEPS=${TOTAL_STEPS}" \
  "SAVE_FREQ=${SAVE_FREQ}" \
  "BATCH_SIZE_PER_GPU=${BATCH_SIZE}" \
  "GLOBAL_BATCH_SIZE=$((BATCH_SIZE * 2))" \
  "NUM_WORKERS=0" \
  "CACHE_LOCALITY_BATCH_SIZE=${CACHE_LOCALITY_BATCH_SIZE}" \
  "CACHE_ACCESS=ascending_no_random_mmap" \
  >"${LOG_DIR}/config.env"

cmd=(
  "${VENV_DIR}/bin/accelerate" launch --multi_gpu --num_processes=2 --mixed_precision=fp16
  -m lerobot.scripts.lerobot_train
  "--config_path=${CONFIG_PATH}"
  --resume=true
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${LOG_RUN_ID}"
  "--steps=${TOTAL_STEPS}"
  "--batch_size=${BATCH_SIZE}"
  --num_workers=0
  --prefetch_factor=2
  --persistent_workers=false
  "--save_freq=${SAVE_FREQ}"
  --save_checkpoint=true
  --log_freq=100
  --eval_freq=0
  --wandb.enable=false
  "--dataset.dino_feature_cache_manifest=${CACHE_MANIFEST}"
  "--dataset.dino_cache_locality_batch_size=${CACHE_LOCALITY_BATCH_SIZE}"
)
printf '%q ' "${cmd[@]}" >"${LOG_DIR}/command.txt"
printf '\n' >>"${LOG_DIR}/command.txt"
printf '[%s] resume seqcache DDP: checkpoint=%s per_gpu_batch=%s global_batch=%s target=%s\n' \
  "$(date '+%F %T')" "${CHECKPOINT_DIR}" "${BATCH_SIZE}" "$((BATCH_SIZE * 2))" "${TOTAL_STEPS}" | tee -a "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
touch "${OUTPUT_DIR}/TRAINING_SUCCEEDED"
