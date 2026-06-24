#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zeno-rp/2027icra"
DEPLOY_DIR="${REPO_ROOT}/scripts/deploy"
LOG_ROOT="${REPO_ROOT}/outputs/logs"
TRAIN_PID="${TRAIN_PID:-1995534}"
PIPELINE_SERVICE="${PIPELINE_SERVICE:-robot-dinov3-act-20260623-r3.service}"
ROBOT4_LOG="${REPO_ROOT}/scripts/train_log/robot4_20260623_act_dinov3_base_dim768/train.log"
SWITCH_LOG="${LOG_ROOT}/robot5_batch32_switch.log"

mkdir -p "${LOG_ROOT}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

{
  echo "[$(timestamp)] waiting for robot4 training pid=${TRAIN_PID}"
  while ps -p "${TRAIN_PID}" >/dev/null 2>&1; do
    sleep 5
  done

  echo "[$(timestamp)] robot4 training process exited"
  if ! grep -q "End of training" "${ROBOT4_LOG}"; then
    echo "[$(timestamp)] robot4 train log does not show successful completion; not starting robot5"
    exit 1
  fi

  echo "[$(timestamp)] stopping old pipeline service: ${PIPELINE_SERVICE}"
  systemctl --user stop "${PIPELINE_SERVICE}" || true
  sleep 2

  echo "[$(timestamp)] starting robot5 with batch_size=32"
  cd "${DEPLOY_DIR}"
  env ROBOTS=robot5 OVERWRITE_DATASET=true BATCH_SIZE=32 \
    "${DEPLOY_DIR}/run_20260623_robot_dinov3_act_pipeline.sh"
} 2>&1 | tee -a "${SWITCH_LOG}"
