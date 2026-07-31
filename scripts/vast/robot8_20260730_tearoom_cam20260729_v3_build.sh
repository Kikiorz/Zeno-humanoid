#!/usr/bin/env bash
# Produce endpoint-accurate, independently XY/yaw-smoothed V3 base labels.
# Visual frames remain hard-linked to the exact 30 Hz rectified source dataset.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_ROOT}/Data/2026_07_30_TeaRoom}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${REPO_ROOT}/Data/lerobot}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260730_tearoom_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_cam20260729_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${LEROBOT_ROOT}/${SOURCE_REPO_ID}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_ROOT}/${DATASET_REPO_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/outputs/analysis/robot8_20260730_tearoom_cam20260729_base_anchor_odom_v3_decoupled_smooth}"
EXCLUDE_BAGS="${EXCLUDE_BAGS:-rosbag2_2026_07_30_20_23_35}"
MIN_SECONDARY_ACCEPTED_FRACTION="${MIN_SECONDARY_ACCEPTED_FRACTION:-0.70}"
MIN_SECONDARY_TV_REDUCTION="${MIN_SECONDARY_TV_REDUCTION:-0.20}"

if [[ ! -x "${VENV_DIR}/bin/python" || ! -d "${RAW_DATA_DIR}" || ! -d "${SOURCE_DATASET}" ]]; then
  printf 'Missing V3 prerequisites. venv=%s raw=%s source=%s\n' \
    "${VENV_DIR}" "${RAW_DATA_DIR}" "${SOURCE_DATASET}" >&2
  exit 1
fi
if [[ -e "${DATASET_ROOT}" ]]; then
  printf 'Refusing to reuse or overwrite a possibly partial V3 dataset: %s\n' "${DATASET_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_HF_HOME:-${REPO_ROOT}/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/.hf_lerobot}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${ANALYSIS_DIR}"

"${VENV_DIR}/bin/python" - "${SOURCE_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
topcam = json.loads((root / "meta" / "topcam_rectification.json").read_text(encoding="utf-8"))
if info.get("fps") != 30:
    raise SystemExit(f"V3 requires the 30 Hz source dataset, got fps={info.get('fps')!r}")
if topcam.get("profile") != "cam_20260729" or topcam.get("head_camera_feature_to_eye") != {"head_cam": "left"}:
    raise SystemExit("V3 source does not have the verified rectified-left-topcam contract")
PY

printf '[%s] deriving 30 Hz endpoint-accurate V3 labels: %s\n' "$(date '+%F %T')" "${DATASET_ROOT}"
"${VENV_DIR}/bin/python" \
  "${REPO_ROOT}/scripts/data_clean/reconstruct_robot8_20260721_base_anchor_odom_v2.py" \
  --source-dataset "${SOURCE_DATASET}" \
  --bag-data-dir "${RAW_DATA_DIR}" \
  --output-dataset "${DATASET_ROOT}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --fps 30 \
  --preview-episodes 3 \
  --video-mode hardlink \
  --dataset-version base_anchor_odom_v3_decoupled_smooth \
  --metadata-dir-name base_anchor_odom_v3_decoupled_smooth \
  --anchor-min-frames 9 \
  --smooth-window 7 \
  --secondary-smooth-window 17 \
  --exclude-bags "${EXCLUDE_BAGS}"

"${VENV_DIR}/bin/python" - "${DATASET_ROOT}" "${MIN_SECONDARY_ACCEPTED_FRACTION}" "${MIN_SECONDARY_TV_REDUCTION}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_fraction = float(sys.argv[2])
min_reduction = float(sys.argv[3])
summary = json.loads((root / "meta" / "base_anchor_odom_v3_decoupled_smooth" / "generation_summary.json").read_text())
if summary.get("version") != "base_anchor_odom_v3_decoupled_smooth":
    raise SystemExit(f"Unexpected V3 version: {summary.get('version')!r}")
if float(summary.get("fps", -1)) != 30:
    raise SystemExit(f"V3 generation did not retain 30 Hz timing: {summary.get('fps')!r}")
secondary = summary.get("secondary_decoupled_smoothing", {})
fraction = float(secondary.get("accepted_action_frame_fraction", -1.0))
reduction = float(secondary.get("velocity_total_variation_reduction_vs_v2", -1.0))
if fraction < min_fraction or reduction < min_reduction:
    raise SystemExit(
        "V3 smoothing quality gate failed: "
        f"accepted_fraction={fraction:.3f} (min={min_fraction:.3f}), "
        f"tv_reduction={reduction:.3f} (min={min_reduction:.3f})"
    )
print(
    "V3 labels verified: "
    f"accepted_secondary_fraction={fraction:.3f}, velocity_tv_reduction={reduction:.3f}",
    flush=True,
)
PY

"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/data_clean/verify_shared_visual_dataset.py" \
  --source-dataset "${SOURCE_DATASET}" \
  --derived-dataset "${DATASET_ROOT}"
