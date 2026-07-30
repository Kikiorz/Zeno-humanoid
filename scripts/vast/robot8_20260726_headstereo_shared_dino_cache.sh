#!/usr/bin/env bash
# A single ~305-GiB four-camera memfd cache is shared by the raw and V3 consumers. Two
# such caches exceed this instance's 483.6-GiB cgroup memory limit.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/third_party/lerobot/.venv}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260726_zeno_h1_auto_cmd_v30_4cam_640x480_headstereo_rectified_crop_lr_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
V3_REPO_ID="${V3_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
V3_DATASET="${V3_DATASET:-${LEROBOT_ROOT}/${V3_REPO_ID}}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/dino_feature_cache}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${CACHE_DIR}/robot8_20260726_headstereo_rectified_lr_shared_dinov3.json}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing LeRobot virtual environment: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_DATASET}" || ! -d "${V3_DATASET}" ]]; then
  printf 'Corrected source/V3 data must exist before cache build. source=%s v3=%s\n' \
    "${SOURCE_DATASET}" "${V3_DATASET}" >&2
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

printf '[%s] building one shared frozen-DINO cache for raw and V3 consumers\n' "$(date '+%F %T')"
exec "${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/build_dino_memfd_feature_cache.py" \
  --dataset-root "${SOURCE_DATASET}" \
  --repo-id "${SOURCE_REPO_ID}" \
  --model vit_base_patch16_dinov3.lvd1689m \
  --cameras head_cam,head_cam_right,left_arm_cam,right_arm_cam \
  --manifest "${CACHE_MANIFEST}" \
  --devices 0,1 \
  --batch-size 64 \
  --video-backend torchcodec \
  --min-free-gib 64 \
  --replace-stale-manifest
