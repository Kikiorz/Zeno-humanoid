#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zeno-rp/2027icra"
DEPLOY_DIR="${REPO_ROOT}/scripts/deploy"
DATA_DIR="${REPO_ROOT}/Data/20260623"
LOG_DIR="${REPO_ROOT}/outputs/logs"
LOG_FILE="${LOG_DIR}/upload_20260623_to_hf.log"
TOKEN_FILE="${HF_TOKEN_FILE:-/tmp/hf_token_20260623}"
REPO_ID="${HF_REPO_ID:-QRP123/20260623}"
CONDA_BIN="${CONDA_BIN:-/home/zeno-rp/miniconda3/bin/conda}"
DEFAULT_PROXY="${HF_UPLOAD_PROXY:-socks5://127.0.0.1:7897}"

mkdir -p "${LOG_DIR}"

for proxy_var in ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy; do
  proxy_value="${!proxy_var:-}"
  if [[ -z "${proxy_value}" ]]; then
    export "${proxy_var}=${DEFAULT_PROXY}"
  elif [[ "${proxy_value}" == socks://* ]]; then
    export "${proxy_var}=socks5://${proxy_value#socks://}"
  fi
done

if [[ ! -s "${TOKEN_FILE}" ]]; then
  echo "Token file not found or empty: ${TOKEN_FILE}" >&2
  exit 1
fi

HF_TOKEN="$(<"${TOKEN_FILE}")"
rm -f "${TOKEN_FILE}"
export HF_TOKEN

cd "${DEPLOY_DIR}"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] upload start"
  echo "repo_id=${REPO_ID}"
  echo "data_dir=${DATA_DIR}"
  du -sh "${DATA_DIR}" || true
  "${CONDA_BIN}" run -n lerobot-qrp312 python upload_folder_to_hf.py \
    --folder "${DATA_DIR}" \
    --repo-id "${REPO_ID}" \
    --repo-type dataset
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] upload finished"
} 2>&1 | tee -a "${LOG_FILE}"
