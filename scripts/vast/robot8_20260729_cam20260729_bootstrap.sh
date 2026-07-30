#!/usr/bin/env bash
# Finite setup for a fresh RTX-5090 Vast instance. The project source stays in
# /workspace; Python packages live in the image-provided /venv/main.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Expected Vast Python environment is missing: %s\n' "${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/third_party/lerobot/pyproject.toml" ]]; then
  printf 'LeRobot source is missing from synced repository: %s\n' "${REPO_ROOT}" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
# Leave wheel selection to the current PyPI default: on this Blackwell host it
# resolves a CUDA>=12.8 PyTorch wheel. Do not pin an obsolete cu124 wheel.
uv pip install --upgrade \
  -e "${REPO_ROOT}/third_party/lerobot[dataset,training,act_dinov2]" \
  rosbags

export PYTHONPATH="${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
"${VENV_DIR}/bin/python" - <<'PY'
import importlib.util
import torch

required = ("lerobot", "rosbags", "cv2", "pyarrow", "timm", "torchcodec", "accelerate")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required modules after bootstrap: " + ", ".join(missing))
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit(f"Expected two CUDA GPUs, got available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
for index in range(2):
    capability = torch.cuda.get_device_capability(index)
    if capability[0] < 10:
        raise SystemExit(f"Unexpected GPU capability for GPU {index}: {capability}")
    x = torch.ones(1, device=f"cuda:{index}")
    assert float(x.item()) == 1.0
print("bootstrap GPU/module smoke passed", torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)])
PY
