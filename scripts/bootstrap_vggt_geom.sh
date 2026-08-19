#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${PROJECT_ROOT}/third_party/VGGT-SLAM"
GEOM_ENV_PREFIX="${VGGT_GEOM_ENV_PREFIX:-/root/autodl-tmp/envs/vggt_geom}"

if [[ ! -f "${UPSTREAM_ROOT}/requirements.txt" ]]; then
  echo "Missing ${UPSTREAM_ROOT}; clone the pinned upstream source first." >&2
  exit 1
fi
if [[ ! -d "${UPSTREAM_ROOT}/third_party/salad" || ! -d "${UPSTREAM_ROOT}/third_party/vggt" ]]; then
  echo "Missing SALAD or VGGT_SPARK under upstream third_party/." >&2
  exit 1
fi

if [[ ! -x "${GEOM_ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "${GEOM_ENV_PREFIX}" python=3.11 pip \
    --override-channels -c https://repo.anaconda.com/pkgs/main
fi

PIP=(
  conda run --no-capture-output -p "${GEOM_ENV_PREFIX}"
  python -m pip install
  --index-url https://pypi.org/simple
  --timeout 60 --retries 10 --no-cache-dir --progress-bar off
)

"${PIP[@]}" torch==2.3.1 torchvision==0.18.1
"${PIP[@]}" \
  numpy==1.26.1 Pillow==12.3.0 opencv-python==4.11.0.86 scipy==1.14.1 \
  matplotlib==3.11.1 termcolor==3.3.0 tqdm==4.70.0 requests==2.34.2 \
  trimesh==5.0.0 lz4==4.4.5 ftfy==6.3.1 regex==2026.7.19 \
  huggingface-hub==1.28.0 einops==0.8.2 safetensors==0.8.0 \
  omegaconf==2.3.1 pytorch-metric-learning==2.9.0 pytorch-lightning==2.6.5 \
  pandas==2.2.3 scikit-image==0.24.0 \
  gtsam-develop==4.3a2.dev202608161900 open3d==0.19.0 viser==0.2.23 \
  PyYAML==6.0.3 wheel

"${PIP[@]}" --no-deps --no-build-isolation \
  -e "${UPSTREAM_ROOT}/third_party/salad" \
  -e "${UPSTREAM_ROOT}/third_party/vggt" \
  -e "${UPSTREAM_ROOT}" \
  -e "${PROJECT_ROOT}"

echo "Geometry environment ready: ${GEOM_ENV_PREFIX}"
echo "No model weights were downloaded by this bootstrap."
echo "Check: conda run --no-capture-output -p ${GEOM_ENV_PREFIX} python -m scripts.run_vggt_geometry --check-only"
