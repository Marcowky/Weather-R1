#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        "EasyR1's latest Dockerfile is based on Python 3.12. "
        f"Current Python is {sys.version.split()[0]}."
    )
PY

python -m pip install --upgrade pip

pip uninstall -y \
    torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    transformer-engine flash-attn apex megatron-core \
    xgboost opencv grpcio

pip install --no-cache-dir \
    "vllm==0.11.0" \
    "torch==2.8.0" \
    "torchvision==0.23.0" \
    "torchaudio==2.8.0" \
    tensordict \
    torchdata \
    "transformers[hf_xet]>=4.51.0" \
    accelerate \
    datasets \
    peft \
    hf-transfer \
    "numpy<2.0.0" \
    "pyarrow>=15.0.0" \
    "grpcio>=1.62.1" \
    "optree>=0.13.0" \
    pandas \
    "ray[default]" \
    codetiming \
    hydra-core \
    pylatexenc \
    qwen-vl-utils \
    wandb \
    liger-kernel \
    mathruler \
    pytest \
    yapf \
    py-spy \
    pre-commit \
    ruff

ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_WHEEL="$(basename "${FLASH_ATTN_URL}")"
FLASH_ATTN_DIR="${FLASH_ATTN_DIR:-.}"

wget -nv -P "${FLASH_ATTN_DIR}" "${FLASH_ATTN_URL}"
pip install --no-cache-dir "${FLASH_ATTN_DIR}/${FLASH_ATTN_WHEEL}"
