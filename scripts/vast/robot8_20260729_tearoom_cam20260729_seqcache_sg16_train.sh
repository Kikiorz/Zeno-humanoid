#!/usr/bin/env bash
# Fast fresh TeaRoom normal-label run: two GPUs, batch 16 per GPU, and strictly
# sequential frozen-DINO cache reads. The cache itself is reused unchanged; no
# video decoding or DINO forward pass occurs during training.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260729_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp32_b16_seqcache_60k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260729_tearoom_topcam_left_cam20260729_normal_dinov3_disk.json}"
DINO_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS:-${REPO_ROOT}/.hf_home/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors}"
TOTAL_STEPS="${TOTAL_STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# It is a multiple of batch 16.  With shuffle_blocks disabled by the sampler,
# either 16 or 2048 yields ascending cache rows; a larger boundary records the
# intent while remaining strictly sequential.
CACHE_LOCALITY_BATCH_SIZE="${CACHE_LOCALITY_BATCH_SIZE:-2048}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${REPO_ROOT}/scripts/train_log}"
LOG_DIR="${TRAIN_LOG_ROOT}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/train.log"

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/accelerate" ]]; then
  printf 'Missing LeRobot environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" || ! -f "${CACHE_MANIFEST}" || ! -f "${DINO_PRETRAINED_WEIGHTS}" ]]; then
  printf 'Missing dataset, cache manifest, or DINO weights.\n' >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite fresh output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
if ! [[ "${TOTAL_STEPS}" =~ ^[0-9]+$ && "${SAVE_FREQ}" =~ ^[0-9]+$ && "${BATCH_SIZE}" =~ ^[0-9]+$ && "${CACHE_LOCALITY_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
  printf 'Step, save-frequency, batch, and locality values must be positive integers.\n' >&2
  exit 1
fi
if (( TOTAL_STEPS < 1 || SAVE_FREQ < 1 || TOTAL_STEPS % SAVE_FREQ != 0 || BATCH_SIZE < 1 || CACHE_LOCALITY_BATCH_SIZE < 1 || CACHE_LOCALITY_BATCH_SIZE % BATCH_SIZE != 0 )); then
  printf 'Invalid step/save/batch/locality relationship.\n' >&2
  exit 1
fi
"${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" "${DATASET_REPO_ID}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("storage") != "disk" or manifest.get("ready") is not True or manifest.get("status") != "ready":
    raise SystemExit("DINO disk cache is not ready")
if manifest.get("dataset", {}).get("repo_id") != sys.argv[2]:
    raise SystemExit("DINO cache was built for a different dataset")
if Path(manifest["cache_path"]).stat().st_size != int(manifest["byte_size"]):
    raise SystemExit("DINO cache byte size changed")
if manifest.get("data_index", {}).get("first") != 0:
    raise SystemExit("Sequential sampler requires absolute cache rows starting at zero")
PY

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
  "RUN_ID=${RUN_ID}" \
  "TOTAL_STEPS=${TOTAL_STEPS}" \
  "SAVE_FREQ=${SAVE_FREQ}" \
  "BATCH_SIZE_PER_GPU=${BATCH_SIZE}" \
  "GLOBAL_BATCH_SIZE=$((BATCH_SIZE * 2))" \
  "NUM_WORKERS=${NUM_WORKERS}" \
  "PREFETCH_FACTOR=${PREFETCH_FACTOR}" \
  "CACHE_LOCALITY_BATCH_SIZE=${CACHE_LOCALITY_BATCH_SIZE}" \
  "CACHE_ACCESS=ascending_no_random_mmap" \
  >"${LOG_DIR}/config.env"

cmd=(
  "${VENV_DIR}/bin/accelerate" launch --multi_gpu --num_processes=2 --mixed_precision=fp16
  -m lerobot.scripts.lerobot_train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  --dataset.video_backend=torchcodec
  --dataset.use_imagenet_stats=true
  "--dataset.dino_feature_cache_manifest=${CACHE_MANIFEST}"
  "--dataset.dino_cache_locality_batch_size=${CACHE_LOCALITY_BATCH_SIZE}"
  --policy.type=act
  --policy.device=cuda
  --policy.use_amp=true
  --policy.push_to_hub=false
  --policy.vision_backbone=dinov2
  --policy.dinov2_model=vit_base_patch16_dinov3.lvd1689m
  --policy.dinov2_pretrained=true
  --policy.dinov2_train_backbone=false
  "--policy.dinov2_pretrained_weights=${DINO_PRETRAINED_WEIGHTS}"
  --policy.n_decoder_layers=7
  --policy.chunk_size=100
  --policy.n_action_steps=100
  --policy.optimizer_lr=1e-5
  --policy.optimizer_lr_backbone=1e-5
  --policy.action_loss_weights='[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${RUN_ID}"
  --seed=1000
  "--steps=${TOTAL_STEPS}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--prefetch_factor=${PREFETCH_FACTOR}"
  --persistent_workers=true
  "--save_freq=${SAVE_FREQ}"
  --save_checkpoint=true
  --log_freq=100
  --eval_freq=0
  --wandb.enable=false
)
printf '%q ' "${cmd[@]}" >"${LOG_DIR}/command.txt"
printf '\n' >>"${LOG_DIR}/command.txt"
printf '[%s] fresh two-GPU sequential-cache training: per_gpu_batch=%s global_batch=%s steps=%s save_freq=%s\n' \
  "$(date '+%F %T')" "${BATCH_SIZE}" "$((BATCH_SIZE * 2))" "${TOTAL_STEPS}" "${SAVE_FREQ}" | tee -a "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
touch "${OUTPUT_DIR}/TRAINING_SUCCEEDED"
printf '[%s] training completed successfully: %s\n' "$(date '+%F %T')" "${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
