#!/usr/bin/env bash
set -euo pipefail

# Fresh, two-GPU ACT+DINOv3 training for the 2026-07-21 data.  Unlike the
# earlier torso-frozen run, all 23 state and action dimensions participate in
# normalization and loss computation.

if (( $# > 0 )); then
  printf 'Configure this script with environment variables, not positional arguments.\n' >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_decoder7_ddp128_100k}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
# This is per GPU; two ranks give the requested global batch size of 128.
BATCH_SIZE="${BATCH_SIZE:-64}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SEED="${SEED:-1000}"
DINOV2_MODEL="${DINOV2_MODEL:-vit_base_patch16_dinov3.lvd1689m}"
DINOV2_PRETRAINED="${DINOV2_PRETRAINED:-true}"
DINOV2_TRAIN_BACKBONE="${DINOV2_TRAIN_BACKBONE:-false}"
DINOV2_PRETRAINED_WEIGHTS="${DINOV2_PRETRAINED_WEIGHTS:-}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${REPO_ROOT}/scripts/train_log}"
LOG_DIR="${TRAIN_LOG_ROOT}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/train.log"
# When non-empty, the dataset supplies frozen-DINO feature maps under the
# training-only observation.dino_features key. The policy then keeps all of
# its normal parameters/checkpoint keys but skips the repeated DINO forward.
DINO_FEATURE_CACHE_MANIFEST="${DINO_FEATURE_CACHE_MANIFEST:-}"

# Be explicit about the full 23-D objective.  This differs from the old
# frozen-torso profiles, whose first two weights were zero.
ACTION_LOSS_WEIGHTS="${ACTION_LOSS_WEIGHTS:-[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}"
# When both are supplied, these replace the static vector above with a
# linearly interpolated, global-step loss curriculum. Values must be JSON-like
# nested lists accepted by Draccus, one 23-D row per listed step.
ACTION_LOSS_WEIGHT_SCHEDULE_STEPS="${ACTION_LOSS_WEIGHT_SCHEDULE_STEPS:-}"
ACTION_LOSS_WEIGHT_SCHEDULE_VALUES="${ACTION_LOSS_WEIGHT_SCHEDULE_VALUES:-}"

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/accelerate" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  printf 'Missing dataset: %s\nRun the all-23D conversion first.\n' "${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -n "${DINO_FEATURE_CACHE_MANIFEST}" && ! -f "${DINO_FEATURE_CACHE_MANIFEST}" ]]; then
  printf 'DINO feature cache manifest does not exist: %s\n' "${DINO_FEATURE_CACHE_MANIFEST}" >&2
  exit 1
fi
if [[ -n "${DINOV2_PRETRAINED_WEIGHTS}" && ! -f "${DINOV2_PRETRAINED_WEIGHTS}" ]]; then
  printf 'Explicit local DINO weights do not exist: %s\n' "${DINOV2_PRETRAINED_WEIGHTS}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite existing fresh-training output: %s\n' "${OUTPUT_DIR}" >&2
  printf 'Use a new RUN_ID, or resume deliberately from a saved checkpoint.\n' >&2
  exit 1
fi
if ! [[ "${TOTAL_STEPS}" =~ ^[0-9]+$ && "${SAVE_FREQ}" =~ ^[0-9]+$ ]]; then
  printf 'TOTAL_STEPS and SAVE_FREQ must be positive decimal integers.\n' >&2
  exit 1
fi
case "${SAVE_CHECKPOINT}" in
  true|false) ;;
  *)
    printf 'SAVE_CHECKPOINT must be true or false, got %s.\n' "${SAVE_CHECKPOINT}" >&2
    exit 1
    ;;
esac
if (( TOTAL_STEPS < 1 || SAVE_FREQ < 1 || TOTAL_STEPS % SAVE_FREQ != 0 )); then
  printf 'SAVE_FREQ must be positive and divide TOTAL_STEPS.\n' >&2
  exit 1
fi
if (( BATCH_SIZE * NUM_PROCESSES != GLOBAL_BATCH_SIZE )); then
  printf 'Per-GPU batch %s x ranks %s is not global batch %s.\n' \
    "${BATCH_SIZE}" "${NUM_PROCESSES}" "${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi
if [[ -n "${ACTION_LOSS_WEIGHT_SCHEDULE_STEPS}" || -n "${ACTION_LOSS_WEIGHT_SCHEDULE_VALUES}" ]]; then
  if [[ -z "${ACTION_LOSS_WEIGHT_SCHEDULE_STEPS}" || -z "${ACTION_LOSS_WEIGHT_SCHEDULE_VALUES}" ]]; then
    printf 'ACTION_LOSS_WEIGHT_SCHEDULE_STEPS and ACTION_LOSS_WEIGHT_SCHEDULE_VALUES must be supplied together.\n' >&2
    exit 1
  fi
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
# Keep the policy loader on the same verified cache as the cache builder,
# rather than inheriting the base image's unrelated global HF_HOME.
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "${LOG_DIR}"
printf '%s\n' \
  "RUN_ID=${RUN_ID}" \
  "DATASET_REPO_ID=${DATASET_REPO_ID}" \
  "DATASET_ROOT=${DATASET_ROOT}" \
  "OUTPUT_DIR=${OUTPUT_DIR}" \
  "TOTAL_STEPS=${TOTAL_STEPS}" \
  "SAVE_FREQ=${SAVE_FREQ}" \
  "SAVE_CHECKPOINT=${SAVE_CHECKPOINT}" \
  "NUM_PROCESSES=${NUM_PROCESSES}" \
  "BATCH_SIZE_PER_GPU=${BATCH_SIZE}" \
  "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" \
  "NUM_WORKERS=${NUM_WORKERS}" \
  "PREFETCH_FACTOR=${PREFETCH_FACTOR}" \
  "DINOV2_MODEL=${DINOV2_MODEL}" \
  "DINOV2_PRETRAINED_WEIGHTS=${DINOV2_PRETRAINED_WEIGHTS:-none}" \
  "DINOV2_TRAIN_BACKBONE=${DINOV2_TRAIN_BACKBONE}" \
  "DINO_FEATURE_CACHE_MANIFEST=${DINO_FEATURE_CACHE_MANIFEST:-none}" \
  "N_DECODER_LAYERS=7" \
  "ACTION_LOSS_WEIGHTS=${ACTION_LOSS_WEIGHTS}" \
  "ACTION_LOSS_WEIGHT_SCHEDULE_STEPS=${ACTION_LOSS_WEIGHT_SCHEDULE_STEPS:-none}" \
  "ACTION_LOSS_WEIGHT_SCHEDULE_VALUES=${ACTION_LOSS_WEIGHT_SCHEDULE_VALUES:-none}" \
  >"${LOG_DIR}/config.env"

cmd=("${VENV_DIR}/bin/accelerate" launch)
if (( NUM_PROCESSES > 1 )); then
  cmd+=(--multi_gpu "--num_processes=${NUM_PROCESSES}")
else
  # Single-card runs must not request Accelerate's multi-GPU launcher: it
  # would otherwise initialize a needless distributed backend and can choose
  # a GPU outside CUDA_VISIBLE_DEVICES on some hosts.
  cmd+=("--num_processes=${NUM_PROCESSES}")
fi
cmd+=(
  --mixed_precision=fp16
  -m lerobot.scripts.lerobot_train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.video_backend=${VIDEO_BACKEND}"
  --dataset.use_imagenet_stats=true
  --policy.type=act
  --policy.device=cuda
  --policy.use_amp=true
  --policy.push_to_hub=false
  --policy.vision_backbone=dinov2
  "--policy.dinov2_model=${DINOV2_MODEL}"
  "--policy.dinov2_pretrained=${DINOV2_PRETRAINED}"
  "--policy.dinov2_train_backbone=${DINOV2_TRAIN_BACKBONE}"
  --policy.n_decoder_layers=7
  --policy.chunk_size=100
  --policy.n_action_steps=100
  "--policy.optimizer_lr=${OPTIMIZER_LR}"
  "--policy.optimizer_lr_backbone=${OPTIMIZER_LR_BACKBONE}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${RUN_ID}"
  "--seed=${SEED}"
  "--steps=${TOTAL_STEPS}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--prefetch_factor=${PREFETCH_FACTOR}"
  "--save_freq=${SAVE_FREQ}"
  "--save_checkpoint=${SAVE_CHECKPOINT}"
  "--log_freq=${LOG_FREQ}"
  --eval_freq=0
  --wandb.enable=false
)

if [[ -n "${DINOV2_PRETRAINED_WEIGHTS}" ]]; then
  cmd+=("--policy.dinov2_pretrained_weights=${DINOV2_PRETRAINED_WEIGHTS}")
fi

if [[ -n "${ACTION_LOSS_WEIGHT_SCHEDULE_STEPS}" ]]; then
  cmd+=(
    "--policy.action_loss_weight_schedule_steps=${ACTION_LOSS_WEIGHT_SCHEDULE_STEPS}"
    "--policy.action_loss_weight_schedule_values=${ACTION_LOSS_WEIGHT_SCHEDULE_VALUES}"
  )
else
  cmd+=("--policy.action_loss_weights=${ACTION_LOSS_WEIGHTS}")
fi

if [[ -n "${DINO_FEATURE_CACHE_MANIFEST}" ]]; then
  cmd+=("--dataset.dino_feature_cache_manifest=${DINO_FEATURE_CACHE_MANIFEST}")
fi

printf '%q ' "${cmd[@]}" >"${LOG_DIR}/command.txt"
printf '\n' >>"${LOG_DIR}/command.txt"
printf '[%s] fresh all-23D training: ranks=%s per_gpu_batch=%s global_batch=%s target=%s save_freq=%s\n' \
  "$(date '+%F %T')" "${NUM_PROCESSES}" "${BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" \
  "${TOTAL_STEPS}" "${SAVE_FREQ}" | tee -a "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
touch "${OUTPUT_DIR}/TRAINING_SUCCEEDED"
printf '[%s] training completed successfully: %s\n' "$(date '+%F %T')" "${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
