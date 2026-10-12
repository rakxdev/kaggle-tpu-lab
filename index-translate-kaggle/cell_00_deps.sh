#!/bin/sh
# CELL 0 — session dependencies. Run FIRST in every Kaggle session.
#
# Installs toolchain, fast Hugging Face download stack, and image support for
# multimodal vision testing.

echo "== 1. apt toolchain =="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential cmake ninja-build git curl ca-certificates libgomp1 > /dev/null 2>&1

for tool in cmake g++ gcc ninja git curl; do
  command -v "$tool" > /dev/null 2>&1 && echo "  $tool : $($tool --version 2>/dev/null | head -1 | head -c 60)" \
    || echo "  !! MISSING: $tool"
done

echo ""
echo "== 2. python dependencies =="
python3 -c "import huggingface_hub; print('  huggingface_hub', huggingface_hub.__version__)" 2>/dev/null \
  || pip install -q huggingface_hub
pip install -q hf_transfer 2>/dev/null && echo "  hf_transfer ready (accelerated downloads)" \
  || echo "  hf_transfer not available — standard download fallback"
python3 -c "import PIL; print('  Pillow', PIL.__version__)" 2>/dev/null \
  || pip install -q pillow

echo ""
echo "DEPS_OK"
