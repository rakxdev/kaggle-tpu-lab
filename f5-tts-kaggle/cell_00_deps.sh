#!/bin/sh
# CELL 0 — session dependencies. Run FIRST in every Kaggle session.
#
# Hardware target: Kaggle 2x T4 (SM75, 15.83 GiB VRAM each).
# Installs system audio codecs (ffmpeg, libsndfile1) and fast download toolchain.

echo "== 1. apt audio libraries & toolchain =="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq ffmpeg libsndfile1 build-essential cmake ninja-build git curl ca-certificates > /dev/null 2>&1

for tool in ffmpeg git curl; do
  command -v "$tool" > /dev/null 2>&1 && echo "  $tool : $($tool -version 2>/dev/null | head -1 | head -c 60 || $tool --version 2>/dev/null | head -1 | head -c 60)" \
    || echo "  !! MISSING: $tool"
done

echo ""
echo "== 2. python download dependencies =="
python3 -c "import huggingface_hub; print('  huggingface_hub', huggingface_hub.__version__)" 2>/dev/null \
  || pip install -q huggingface_hub
pip install -q hf_transfer 2>/dev/null && echo "  hf_transfer ready (accelerated downloads)" \
  || echo "  hf_transfer not available — standard download fallback"
pip install -q soundfile cached_path 2>/dev/null && echo "  soundfile & cached_path ready"

echo ""
echo "DEPS_OK"
