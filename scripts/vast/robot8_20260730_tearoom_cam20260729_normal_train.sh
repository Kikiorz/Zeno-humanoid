#!/usr/bin/env bash
# GPU 0 normal-label ACT training. DINO is frozen and read from the shared cache.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260730_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260730_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_b32_100k_seqcache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260730_tearoom_topcam_left_cam20260729_shared_dinov3_disk.json}"
CACHE_FILE="${CACHE_FILE:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260730_tearoom_topcam_left_cam20260729_shared_dinov3.f16}"
DINO_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS:-${REPO_ROOT}/.hf_home/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors}"

if [[ ! -d "${DATASET_ROOT}" || ! -x "${VENV_DIR}/bin/python" || ! -f "${DINO_PRETRAINED_WEIGHTS}" ]]; then
  printf 'Missing normal-training prerequisites. dataset=%s venv=%s weights=%s\n' \
    "${DATASET_ROOT}" "${VENV_DIR}" "${DINO_PRETRAINED_WEIGHTS}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite normal training output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
"${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" "${CACHE_FILE}" "${DATASET_REPO_ID}" "${DATASET_ROOT}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
cache_file = Path(sys.argv[2])
repo_id = sys.argv[3]
dataset_root = Path(sys.argv[4])
repo_root = Path(sys.argv[5])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("storage") != "disk" or manifest.get("ready") is not True or manifest.get("status") != "ready":
    raise SystemExit("Shared DINO disk cache is not ready")
if manifest.get("dataset", {}).get("repo_id") != repo_id:
    raise SystemExit("DINO cache was built for another visual dataset")
if Path(manifest.get("cache_path", "")) != cache_file:
    raise SystemExit("DINO cache manifest path differs from the expected 07-30 cache")
if cache_file.stat().st_size != int(manifest["byte_size"]):
    raise SystemExit("DINO cache byte size changed")
if manifest.get("camera_keys") != [
    "observation.images.head_cam",
    "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
]:
    raise SystemExit("DINO cache does not have the three-camera rectified-left-topcam order")
sys.path.insert(0, str(repo_root / "scripts"))
import build_dino_memfd_feature_cache as cache_shared

validation = cache_shared.validate_dataset(
    dataset_root,
    repo_id,
    (
        "observation.images.head_cam",
        "observation.images.left_arm_cam",
        "observation.images.right_arm_cam",
    ),
)
cached_dataset = manifest.get("dataset", {})
if cached_dataset.get("info_sha256") != validation.info_sha256:
    raise SystemExit("DINO cache info.json provenance mismatch")
if cached_dataset.get("source_fingerprint") != validation.source_fingerprint:
    raise SystemExit("DINO cache source fingerprint mismatch")
index = manifest.get("data_index", {})
if index.get("first") != 0 or index.get("count") != validation.total_frames:
    raise SystemExit("DINO cache does not cover the exact source index range")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export REPO_ROOT VENV_DIR DATASET_REPO_ID DATASET_ROOT RUN_ID OUTPUT_DIR
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export NUM_PROCESSES=1
export BATCH_SIZE=32
export GLOBAL_BATCH_SIZE=32
export NUM_WORKERS="${NUM_WORKERS:-12}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export DINO_CACHE_LOCALITY_BATCH_SIZE="${DINO_CACHE_LOCALITY_BATCH_SIZE:-2048}"
export VIDEO_BACKEND=torchcodec
export DINOV2_TRAIN_BACKBONE=false
export DINOV2_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS}"
export DINO_FEATURE_CACHE_MANIFEST="${CACHE_MANIFEST}"
export ACTION_LOSS_WEIGHTS='[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'

exec bash "${REPO_ROOT}/scripts/train_robot8_20260721_act_dinov3_ddp128_all23_100k.sh"
