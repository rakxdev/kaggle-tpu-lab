#!/bin/sh
# CELL 0 — session dependencies. Run FIRST, every session.
#
# Cost: ~1 min, no GPU. Installs the toolchain the llama.cpp fork builds with
# and the download stack. Idempotent: re-running is free and quiet.
#
# Note: ninja (if present) roughly halves the cmake build time — cell_02
# picks it up automatically.

echo "== apt packages =="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential cmake ninja-build git curl ca-certificates \
  > /dev/null 2>&1
for t in cmake g++ gcc ninja git curl; do
  command -v "$t" > /dev/null 2>&1 && echo "  $t : $($t --version 2>/dev/null | head -1 | head -c 60)" \
    || echo "  !! MISSING: $t"
done

echo ""
echo "== python side (downloads) =="
python3 -c "import huggingface_hub; print('  huggingface_hub', huggingface_hub.__version__)" \
  || pip install -q huggingface_hub
pip install -q hf_transfer 2>/dev/null && echo "  hf_transfer installed (fast downloads)" \
  || echo "  hf_transfer not available — downloads will use the standard path"

echo ""
echo "DEPS_OK"