#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DetailPair/DetailPair_v0.2.4
export PIP_CACHE_DIR=/workspace/.cache/pip
export YOLO_CONFIG_DIR=/workspace/DetailPair/DetailPair_v0.2.4/local_settings
export MPLCONFIGDIR=/workspace/.cache/matplotlib
python3 -m venv /workspace/venvs/detailpair
/workspace/venvs/detailpair/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
/workspace/venvs/detailpair/bin/python -m pip install -r requirements.txt
/workspace/venvs/detailpair/bin/python -m pip check
