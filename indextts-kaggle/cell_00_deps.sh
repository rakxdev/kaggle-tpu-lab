#!/bin/sh
# CELL 0 — Session dependencies. Run FIRST in every Kaggle session.
#
# Hardware target: Kaggle 2x T4 (SM75, 15.83 GiB VRAM each).
# Installs system audio codecs (ffmpeg, libsndfile1) and fast download toolchain.

echo "== 1. apt audio codecs & toolchain =="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq ffmpeg libsndfile1 build-essential cmake ninja-build git curl ca-certificates sox libsox-dev > /dev/null 2>&1

for tool in ffmpeg git curl sox; do
  command -v "$tool" > /dev/null 2>&1 && echo "  $tool : $($tool -version 2>/dev/null | head -1 | head -c 60 || $tool --version 2>/dev/null | head -1 | head -c 60)" \
    || echo "  !! MISSING: $tool"
done

echo ""
echo "== 2. python download & uv package manager =="
pip install -q -U uv "huggingface_hub<2.0,>=1.16.0" hf_transfer
echo "  uv and huggingface_hub ready"

echo ""
echo "DEPS_OK"
