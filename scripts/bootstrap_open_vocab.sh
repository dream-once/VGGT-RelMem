#!/usr/bin/env bash
set -euo pipefail

OPEN_VOCAB_PREFIX="${OPEN_VOCAB_ENV_PREFIX:-/root/autodl-tmp/envs/open_vocab}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PE_ROOT="${PROJECT_ROOT}/third_party/VGGT-SLAM/third_party/perception_models"
SAM3_ROOT="${PROJECT_ROOT}/third_party/VGGT-SLAM/third_party/sam3"
PE_COMMIT="3e352cca660658d4b5c90f42a7808b11469e4c66"
SAM3_COMMIT="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"

test "$(git -C "${PE_ROOT}" rev-parse HEAD)" = "${PE_COMMIT}"
test "$(git -C "${SAM3_ROOT}" rev-parse HEAD)" = "${SAM3_COMMIT}"

if [[ ! -x "${OPEN_VOCAB_PREFIX}/bin/python" ]]; then
  conda create -y --override-channels \
    -c https://repo.anaconda.com/pkgs/main -p "${OPEN_VOCAB_PREFIX}" python=3.12 pip
fi

conda run --no-capture-output -p "${OPEN_VOCAB_PREFIX}" \
  python -m pip install --upgrade pip "setuptools<81" wheel
conda run --no-capture-output -p "${OPEN_VOCAB_PREFIX}" \
  python -m pip install \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
conda run --no-capture-output -p "${OPEN_VOCAB_PREFIX}" \
  python -m pip install \
    numpy==1.26.4 pillow==11.0.0 timm==1.0.17 einops==0.8.1 \
    ftfy==6.1.1 regex iopath==0.1.10 huggingface_hub typing_extensions tqdm pycocotools psutil
conda run --no-capture-output -p "${OPEN_VOCAB_PREFIX}" \
  python -m pip install --no-deps -e "${PE_ROOT}" -e "${SAM3_ROOT}"

conda run --no-capture-output -p "${OPEN_VOCAB_PREFIX}" python -c \
  "import torch, torchvision, timm; import core.vision_encoder.pe; import sam3.model_builder; print({'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.cuda.is_available()})"
