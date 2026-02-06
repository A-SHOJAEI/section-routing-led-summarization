#!/usr/bin/env bash
set -euo pipefail

PIP_BIN="${1:-pip}"

# Install a pinned torch, preferring CUDA wheels if available.
# Tries several common CUDA indices and falls back to PyPI (often CPU-only).
TORCH_VERSION="2.10.0"

try_install() {
  local index_url="$1"
  echo "[install_torch] Trying torch==${TORCH_VERSION} from: ${index_url}"
  "${PIP_BIN}" install --no-cache-dir --index-url "${index_url}" "torch==${TORCH_VERSION}" && return 0
  return 1
}

if command -v nvidia-smi >/dev/null 2>&1; then
  # Driver is expected to be backward compatible with CUDA 12.x wheels.
  try_install "https://download.pytorch.org/whl/cu126" || \
  try_install "https://download.pytorch.org/whl/cu124" || \
  try_install "https://download.pytorch.org/whl/cu121" || \
  try_install "https://download.pytorch.org/whl/cpu" || \
  "${PIP_BIN}" install --no-cache-dir "torch==${TORCH_VERSION}"
else
  echo "[install_torch] No nvidia-smi detected; installing CPU torch==${TORCH_VERSION}"
  try_install "https://download.pytorch.org/whl/cpu" || "${PIP_BIN}" install --no-cache-dir "torch==${TORCH_VERSION}"
fi

echo "[install_torch] Verifying torch import..."
"${PIP_BIN%/*}/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

