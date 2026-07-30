#!/usr/bin/env bash
# Consumer-only V3-label training on GPU 1. The visual DINO cache is the
# exact same durable file as the raw-label run; only labels/loss schedule vary.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260729_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/Data/lerobot/${SOURCE_REPO_ID}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260729_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${REPO_ROOT}/outputs/dino_feature_cache/robot8_20260729_topcam_left_cam20260729_shared_dinov3_disk.json}"
DINO_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS:-${REPO_ROOT}/.hf_home/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors}"

if [[ ! -d "${DATASET_ROOT}" || ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Missing V3 dataset or environment. dataset=%s venv=%s\n' "${DATASET_ROOT}" "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DINO_PRETRAINED_WEIGHTS}" ]]; then
  printf 'Missing explicit local DINOv3 weights: %s\n' "${DINO_PRETRAINED_WEIGHTS}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite V3 training output: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
"${VENV_DIR}/bin/python" - "${CACHE_MANIFEST}" "${SOURCE_REPO_ID}" "${SOURCE_DATASET}" <<'PY'
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
    raise SystemExit("DINO cache was built for a different visual source dataset")
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
    source_dataset = Path(sys.argv[3])
    topcam = source_dataset / "meta" / "topcam_rectification.json"
    digest = hashlib.sha256(topcam.read_bytes()).hexdigest()
    if payload.get("topcam_rectification", {}).get("sha256") != digest:
        raise SystemExit("DINO cache was not built from this exact topcam rectification contract")
    sys.path.insert(0, str(Path(os.environ["REPO_ROOT"]) / "scripts"))
    import build_dino_memfd_feature_cache as cache_shared

    validation = cache_shared.validate_dataset(
        source_dataset,
        sys.argv[2],
        (
            "observation.images.head_cam",
            "observation.images.left_arm_cam",
            "observation.images.right_arm_cam",
        ),
    )
    cached_dataset = payload.get("dataset", {})
    if cached_dataset.get("info_sha256") != validation.info_sha256:
        raise SystemExit("DINO cache info.json provenance does not match the source dataset")
    if cached_dataset.get("source_fingerprint") != validation.source_fingerprint:
        raise SystemExit("DINO cache source fingerprint does not match the source dataset")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export REPO_ROOT VENV_DIR DATASET_REPO_ID DATASET_ROOT RUN_ID OUTPUT_DIR
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export NUM_PROCESSES=1
export BATCH_SIZE=32
export GLOBAL_BATCH_SIZE=32
export NUM_WORKERS="${NUM_WORKERS:-4}"
export VIDEO_BACKEND=torchcodec
export DINOV2_TRAIN_BACKBONE=false
export DINOV2_PRETRAINED_WEIGHTS="${DINO_PRETRAINED_WEIGHTS}"
export DINO_FEATURE_CACHE_MANIFEST="${CACHE_MANIFEST}"
export ACTION_LOSS_WEIGHT_SCHEDULE_STEPS='[0,25000,60000,75000,90000,100000]'
export ACTION_LOSS_WEIGHT_SCHEDULE_VALUES="$(${VENV_DIR}/bin/python - <<'PY'
import json
groups = [(3.0, 1.0), (3.0, 1.0), (1.0, 3.0), (1.0, 3.0), (1.0, 1.0), (1.0, 1.0)]
print(json.dumps([[upper / 20.0] * 20 + [base / 3.0] * 3 for upper, base in groups], separators=(",", ":")))
PY
)"

exec bash "${REPO_ROOT}/scripts/train_robot8_20260721_act_dinov3_ddp128_all23_100k.sh"
