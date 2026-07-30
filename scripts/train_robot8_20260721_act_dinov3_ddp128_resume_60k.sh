#!/usr/bin/env bash
set -euo pipefail

# Two-GPU resume preset: 64 samples per GPU, global batch 128, and checkpoints
# every 5k optimizer steps through the global 100k target.
if (( $# > 0 )); then
  printf 'Configure this script with environment variables, not positional arguments.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BATCH_SIZE="${BATCH_SIZE:-64}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export TOTAL_STEPS="${TOTAL_STEPS:-100000}"
export RESUME_STEP="${RESUME_STEP:-060000}"

exec "${SCRIPT_DIR}/train_robot8_20260721_act_dinov3_ddp64_resume_60k.sh"
