#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zeno-rp/2027icra"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Data/20260623}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/outputs/logs}"
CONDA_BIN="${CONDA_BIN:-/home/zeno-rp/miniconda3/bin/conda}"

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

for proxy_var in ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy; do
  proxy_value="${!proxy_var:-}"
  if [[ "${proxy_value}" == socks://* ]]; then
    export "${proxy_var}=socks5://${proxy_value#socks://}"
  fi
done

FPS="${FPS:-20}"
IMG_SIZE="${IMG_SIZE:-224}"
VCODEC="${VCODEC:-}"
OVERWRITE_DATASET="${OVERWRITE_DATASET:-false}"

DINO_MODEL="${DINO_MODEL:-vit_base_patch16_dinov3.lvd1689m}"
DINOV2_PRETRAINED="${DINOV2_PRETRAINED:-true}"
DINOV2_TRAIN_BACKBONE="${DINOV2_TRAIN_BACKBONE:-false}"

DEVICE="${DEVICE:-cuda}"
USE_AMP="${USE_AMP:-true}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-200}"

ACT_DIM_MODEL="${ACT_DIM_MODEL:-768}"
ACT_N_HEADS="${ACT_N_HEADS:-12}"
ACT_DIM_FEEDFORWARD="${ACT_DIM_FEEDFORWARD:-4096}"
ACT_N_ENCODER_LAYERS="${ACT_N_ENCODER_LAYERS:-6}"
ACT_N_VAE_ENCODER_LAYERS="${ACT_N_VAE_ENCODER_LAYERS:-4}"
ENABLE_IMAGE_TRANSFORMS="${ENABLE_IMAGE_TRANSFORMS:-true}"

mkdir -p "${LOG_ROOT}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

convert_robot() {
  local robot="$1"
  local dataset="$2"
  local task="$3"
  local log_file="${LOG_ROOT}/${dataset}_convert.log"

  local overwrite_args=()
  if [[ "${OVERWRITE_DATASET}" == "true" ]]; then
    overwrite_args+=(--overwrite)
  fi
  local vcodec_args=()
  if [[ -n "${VCODEC}" ]]; then
    vcodec_args+=(--vcodec "${VCODEC}")
  fi

  echo "[$(timestamp)] convert ${robot} -> ${dataset}" | tee -a "${log_file}"
  "${CONDA_BIN}" run -n lerobot-qrp312 python "${REPO_ROOT}/scripts/data_convert/convert_zeno_h1_v30.py" \
    --data-dir "${DATA_ROOT}/${robot}" \
    --output-dir "${LEROBOT_ROOT}" \
    --repo-name "${dataset}" \
    --task "${task}" \
    --fps "${FPS}" \
    --img-size "${IMG_SIZE}" \
    "${vcodec_args[@]}" \
    "${overwrite_args[@]}" 2>&1 | tee -a "${log_file}"
}

train_robot() {
  local robot="$1"
  local dataset="$2"
  local run_id="${robot}_20260623_act_dinov3_base_dim${ACT_DIM_MODEL}"
  local log_file="${LOG_ROOT}/${run_id}.log"

  echo "[$(timestamp)] train ${robot} from ${dataset} -> ${run_id}" | tee -a "${log_file}"
  "${CONDA_BIN}" run -n lerobot-qrp312 env \
    DATASET_REPO_ID="${dataset}" \
    DATASET_ROOT="${LEROBOT_ROOT}/${dataset}" \
    RUN_ID="${run_id}" \
    JOB_NAME="${run_id}" \
    OUTPUT_DIR="${REPO_ROOT}/outputs/train/${run_id}" \
    TRAIN_LOG_DIR="${REPO_ROOT}/scripts/train_log/${run_id}" \
    DEVICE="${DEVICE}" \
    USE_AMP="${USE_AMP}" \
    STEPS="${STEPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    SAVE_FREQ="${SAVE_FREQ}" \
    LOG_FREQ="${LOG_FREQ}" \
    DINOV2_MODEL="${DINO_MODEL}" \
    DINOV2_PRETRAINED="${DINOV2_PRETRAINED}" \
    DINOV2_TRAIN_BACKBONE="${DINOV2_TRAIN_BACKBONE}" \
    bash "${REPO_ROOT}/scripts/train_humanmoid_pick_act_dinov2.sh" \
      --policy.dim_model="${ACT_DIM_MODEL}" \
      --policy.n_heads="${ACT_N_HEADS}" \
      --policy.dim_feedforward="${ACT_DIM_FEEDFORWARD}" \
      --policy.n_encoder_layers="${ACT_N_ENCODER_LAYERS}" \
      --policy.n_vae_encoder_layers="${ACT_N_VAE_ENCODER_LAYERS}" \
      --dataset.image_transforms.enable="${ENABLE_IMAGE_TRANSFORMS}" 2>&1 | tee -a "${log_file}"
}

main() {
  echo "[$(timestamp)] pipeline start"
  echo "DINO_MODEL=${DINO_MODEL}"
  echo "ACT dim=${ACT_DIM_MODEL}, heads=${ACT_N_HEADS}, ffn=${ACT_DIM_FEEDFORWARD}, encoder_layers=${ACT_N_ENCODER_LAYERS}"
  echo "batch_size=${BATCH_SIZE}, steps=${STEPS}, transforms=${ENABLE_IMAGE_TRANSFORMS}"

  for robot in ${ROBOTS:-robot4 robot5}; do
    case "${robot}" in
      robot4)
        convert_robot "robot4" "robot4_20260623_zeno_h1_auto_cmd_v30" "robot4_pick"
        train_robot "robot4" "robot4_20260623_zeno_h1_auto_cmd_v30"
        ;;
      robot5)
        convert_robot "robot5" "robot5_20260623_zeno_h1_auto_cmd_v30" "robot5_pick"
        train_robot "robot5" "robot5_20260623_zeno_h1_auto_cmd_v30"
        ;;
      *)
        echo "Unknown robot: ${robot}" >&2
        exit 2
        ;;
    esac
  done

  echo "[$(timestamp)] pipeline complete"
}

main "$@"
