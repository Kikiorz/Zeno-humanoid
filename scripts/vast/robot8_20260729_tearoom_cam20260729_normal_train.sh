#!/usr/bin/env bash
# Two-GPU normal-label TeaRoom ACT run. Each rank receives batch 32; the
# effective global batch is therefore 64. Six loaders per rank are the
# conservative first tuning point for the disk-backed DINO cache. The requested
# normal-label pass runs for 60k optimizer steps and writes a durable
# checkpoint every 10k steps.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260729_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp64_b32_w6_60k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260729_tearoom_topcam_left_cam20260729_normal_dinov3_disk.json}"
DINO_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS:-${REPO_ROOT}/.hf_home/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors}"

if [[ ! -d "${DATASET_ROOT}" || ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing source dataset or environment. dataset=%s venv=%s\n' \
    "${DATASET_ROOT}" "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DINO_PRETRAINED_WEIGHTS}" ]]; then
  printf 'Missing explicit local DINOv3 weights: %s\n' "${DINO_PRETRAINED_WEIGHTS}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite normal training output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
"${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" "${DATASET_REPO_ID}" "${DATASET_ROOT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("storage") != "disk" or payload.get("ready") is not True or payload.get("status") != "ready":
    raise SystemExit(f"durable DINO cache is not ready: storage={payload.get('storage')!r} status={payload.get('status')!r}")
if payload.get("dataset", {}).get("repo_id") != sys.argv[2]:
    raise SystemExit("DINO cache was built for a different source dataset")
cache = Path(str(payload["cache_path"]))
if cache.stat().st_size != int(payload["byte_size"]):
    raise SystemExit("DINO disk-cache byte size changed")
if payload.get("camera_keys") != [
    "observation.images.head_cam", "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
]:
    raise SystemExit("DINO cache camera order is not the left-topcam three-camera contract")
index = payload.get("data_index", {})
if index.get("first") != 0 or index.get("count") != payload.get("dataset", {}).get("total_frames"):
    raise SystemExit("DINO disk cache does not cover every absolute source frame")
if os.environ.get("REQUIRE_TOPCAM_CACHE_PROVENANCE") == "1":
    dataset_root = Path(sys.argv[3])
    topcam = dataset_root / "meta" / "topcam_rectification.json"
    digest = hashlib.sha256(topcam.read_bytes()).hexdigest()
    if payload.get("topcam_rectification", {}).get("sha256") != digest:
        raise SystemExit("DINO cache was not built from this exact topcam rectification contract")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export REPO_ROOT VENV_DIR DATASET_REPO_ID DATASET_ROOT RUN_ID OUTPUT_DIR
export TOTAL_STEPS="${TOTAL_STEPS:-60000}"
export SAVE_FREQ="${SAVE_FREQ:-10000}"
export NUM_PROCESSES=2
export BATCH_SIZE=32
export GLOBAL_BATCH_SIZE=64
export NUM_WORKERS="${NUM_WORKERS:-6}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export VIDEO_BACKEND=torchcodec
export DINOV2_TRAIN_BACKBONE=false
export DINOV2_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS}"
export DINO_FEATURE_CACHE_MANIFEST="${CACHE_MANIFEST}"
export ACTION_LOSS_WEIGHTS='[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'

exec bash "${REPO_ROOT}/scripts/train_robot8_20260721_act_dinov3_ddp128_all23_100k.sh"
