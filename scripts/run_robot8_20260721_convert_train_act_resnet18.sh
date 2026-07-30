#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_REPO_ID="${DATASET_REPO_ID:-robot8_20260721_zeno_h1_auto_cmd_v30_center_crop_2of3_224x224}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/Data/lerobot/${DATASET_REPO_ID}}"
RUN_ID="${RUN_ID:-robot8_20260721_act_resnet18_60k_224x224_crop2of3}"
TRAIN_OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"

if python3 - "${DATASET_ROOT}/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
if not info_path.is_file():
    raise SystemExit(1)
info = json.loads(info_path.read_text())
complete = (
    info.get("fps") == 20
    and info.get("total_episodes") == 35
    and info.get("total_frames") == 49876
)
raise SystemExit(0 if complete else 1)
PY
then
  echo "Dataset already complete; skipping conversion: ${DATASET_ROOT}"
else
  bash "${REPO_ROOT}/scripts/convert_robot8_20260721_20hz.sh"
fi

if python3 - "${TRAIN_OUTPUT_DIR}/checkpoints/060000" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
step_path = checkpoint / "training_state/training_step.json"
required = (
    checkpoint / "pretrained_model/model.safetensors",
    checkpoint / "pretrained_model/config.json",
    checkpoint / "pretrained_model/train_config.json",
    checkpoint / "training_state/optimizer_state.safetensors",
    step_path,
)
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
step = json.loads(step_path.read_text()).get("step")
raise SystemExit(0 if step == 60000 else 1)
PY
then
  echo "Training already complete; skipping: ${TRAIN_OUTPUT_DIR}"
else
  bash "${REPO_ROOT}/scripts/train_robot8_20260721_act_resnet18_60k.sh" "$@"
fi
