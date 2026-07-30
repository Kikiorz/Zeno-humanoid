#!/usr/bin/env bash
# Build one durable disk-backed DINOv3 cache shared by the raw and V3 runs.
# A full 3-camera cache is ~98 GiB; durable mmap avoids repeated backbone work.
# RAM limit, so do not replace this with the memfd cache server.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260729_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
V3_REPO_ID="${V3_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
V3_DATASET="${V3_DATASET:-${LEROBOT_ROOT}/${V3_REPO_ID}}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/dino_feature_cache}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${CACHE_DIR}/robot8_20260729_topcam_left_cam20260729_shared_dinov3_disk.json}"
CACHE_FILE="${CACHE_FILE:-${CACHE_DIR}/robot8_20260729_topcam_left_cam20260729_shared_dinov3.f16}"
DINO_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS:-${REPO_ROOT}/.hf_home/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_DATASET}" || ! -d "${V3_DATASET}" ]]; then
  printf 'Source/V3 datasets must exist before DINO cache build. source=%s v3=%s\n' \
    "${SOURCE_DATASET}" "${V3_DATASET}" >&2
  exit 1
fi
if [[ ! -f "${DINO_PRETRAINED_WEIGHTS}" ]]; then
  printf 'Missing explicit local DINOv3 weights: %s\n' "${DINO_PRETRAINED_WEIGHTS}" >&2
  exit 1
fi
if [[ -e "${CACHE_MANIFEST}" || -e "${CACHE_FILE}" ]]; then
  printf 'Refusing to overwrite cache artifacts. manifest=%s cache=%s\n' \
    "${CACHE_MANIFEST}" "${CACHE_FILE}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${CACHE_DIR}"

"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_clean/verify_shared_visual_dataset.py" \
  --source-dataset "${SOURCE_DATASET}" \
  --derived-dataset "${V3_DATASET}"

printf '[%s] building shared durable frozen-DINO cache\n' "$(date '+%F %T')"
exec "${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/build_dino_disk_feature_cache.py" \
  --dataset-root "${SOURCE_DATASET}" \
  --repo-id "${SOURCE_REPO_ID}" \
  --model vit_base_patch16_dinov3.lvd1689m \
  --pretrained-weights "${DINO_PRETRAINED_WEIGHTS}" \
  --cameras head_cam,left_arm_cam,right_arm_cam \
  --manifest "${CACHE_MANIFEST}" \
  --cache-file "${CACHE_FILE}" \
  --devices 0,1 \
  --batch-size 64 \
  --video-backend torchcodec \
  --min-free-disk-gib 64 \
  --flush-every-batches 1
