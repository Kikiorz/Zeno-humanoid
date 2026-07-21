#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot-qrp312}"

DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_frozen_lift_waist}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${REPO_ROOT}/scripts/train_log}"

BASE_RUN_ID="${BASE_RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_frozen_lift_waist_decoder7_10k}"
BASE3X_RUN_ID="${BASE3X_RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_frozen_lift_waist_base3x_decoder7_10k}"
TRAIN_VARIANT="${TRAIN_VARIANT:-all}" # all, normal, or base3x

DEVICE="${DEVICE:-cuda}"
USE_AMP="${USE_AMP:-true}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SEED="${SEED:-1000}"

# A physical batch of two has ample room on this 24 GiB GPU. The three cameras
# share one frozen DINOv3 backbone, but each 640x480 image still needs a
# forward pass and its resulting visual tokens still consume activation memory.
BATCH_SIZE="${BATCH_SIZE:-2}"

DINOV2_MODEL="${DINOV2_MODEL:-vit_base_patch16_dinov3.lvd1689m}"
DINOV2_PRETRAINED="${DINOV2_PRETRAINED:-true}"
DINOV2_TRAIN_BACKBONE="${DINOV2_TRAIN_BACKBONE:-false}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"

NORMAL_ACTION_LOSS_WEIGHTS='[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'
BASE3X_ACTION_LOSS_WEIGHTS='[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3]'
FROZEN_FIELDS="torso_lift,torso_waist"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  printf 'Dataset does not exist: %s\nRun conversion first.\n' "${DATASET_ROOT}" >&2
  exit 1
fi
case "${TRAIN_VARIANT}" in
  all|normal|base3x) ;;
  *)
    printf 'TRAIN_VARIANT must be all, normal, or base3x; got %s\n' "${TRAIN_VARIANT}" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
# DINOv3 weights are already cached locally.  The inherited socks ALL_PROXY is
# not supported by this environment's HTTP client and is unnecessary here.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy FTP_PROXY ftp_proxy

runner=()
if [[ -n "${CONDA_ENV}" ]]; then
  runner=(conda run --no-capture-output -n "${CONDA_ENV}")
fi

run_train() {
  local run_id="$1"
  local loss_profile="$2"
  local action_loss_weights="$3"
  local output_dir="${REPO_ROOT}/outputs/train/${run_id}"
  local train_log_dir="${TRAIN_LOG_ROOT}/${run_id}"
  local log_path="${train_log_dir}/train.log"

  if [[ -e "${output_dir}" ]]; then
    printf 'Refusing to overwrite existing output: %s\n' "${output_dir}" >&2
    printf 'Use a new RUN_ID, or deliberately remove it after checking its contents.\n' >&2
    exit 1
  fi
  mkdir -p "${train_log_dir}"

  printf '%s\n' \
    "RUN_ID=${run_id}" \
    "LOSS_PROFILE=${loss_profile}" \
    "DATASET_REPO_ID=${DATASET_REPO_ID}" \
    "DATASET_ROOT=${DATASET_ROOT}" \
    "OUTPUT_DIR=${output_dir}" \
    "STEPS=${STEPS}" \
    "SAVE_FREQ=${SAVE_FREQ}" \
    "BATCH_SIZE=${BATCH_SIZE}" \
    "N_DECODER_LAYERS=7" \
    "DINOV2_MODEL=${DINOV2_MODEL}" \
    "DINOV2_TRAIN_BACKBONE=${DINOV2_TRAIN_BACKBONE}" \
    "FROZEN_FIELDS=${FROZEN_FIELDS}" \
    "ACTION_LOSS_WEIGHTS=${action_loss_weights}" \
    >"${train_log_dir}/config.env"

  printf '%s\n' \
    "# ${run_id}" \
    "" \
    "- Dataset state/action fields torso_lift and torso_waist are raw-zeroed." \
    "- Their action-loss weights are zero, so they do not train the policy." \
    "- Physical batch size is ${BATCH_SIZE}." \
    "- DINOv3 ViT-B/16 backbone is frozen; ACT decoder has 7 layers." \
    "- ${loss_profile}." \
    >"${train_log_dir}/notes.md"

  local cmd=(
    "${runner[@]}"
    python3 -m lerobot.scripts.lerobot_train
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend="${VIDEO_BACKEND}"
    --dataset.use_imagenet_stats=true
    --policy.type=act
    --policy.device="${DEVICE}"
    --policy.use_amp="${USE_AMP}"
    --policy.push_to_hub=false
    --policy.vision_backbone=dinov2
    --policy.dinov2_model="${DINOV2_MODEL}"
    --policy.dinov2_pretrained="${DINOV2_PRETRAINED}"
    --policy.dinov2_train_backbone="${DINOV2_TRAIN_BACKBONE}"
    --policy.n_decoder_layers=7
    --policy.chunk_size=100
    --policy.n_action_steps=100
    --policy.action_loss_weights="${action_loss_weights}"
    --policy.optimizer_lr="${OPTIMIZER_LR}"
    --policy.optimizer_lr_backbone="${OPTIMIZER_LR_BACKBONE}"
    --output_dir="${output_dir}"
    --job_name="${run_id}"
    --seed="${SEED}"
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --prefetch_factor="${PREFETCH_FACTOR}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --eval_freq=0
    --wandb.enable=false
  )

  printf '%q ' "${cmd[@]}" >"${train_log_dir}/command.txt"
  printf '\n' >>"${train_log_dir}/command.txt"
  printf '[%s] starting %s (%s)\n' "$(date '+%F %T')" "${run_id}" "${loss_profile}" | tee "${log_path}"
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
}

if [[ "${TRAIN_VARIANT}" == "all" || "${TRAIN_VARIANT}" == "normal" ]]; then
  run_train \
    "${BASE_RUN_ID}" \
    "normal frozen training: active joints use 1x L1 action loss" \
    "${NORMAL_ACTION_LOSS_WEIGHTS}"
fi

if [[ "${TRAIN_VARIANT}" == "all" || "${TRAIN_VARIANT}" == "base3x" ]]; then
  run_train \
    "${BASE3X_RUN_ID}" \
    "base-emphasized training: base_vx/base_vy/base_rotation each use 3x L1 action loss" \
    "${BASE3X_ACTION_LOSS_WEIGHTS}"
fi
