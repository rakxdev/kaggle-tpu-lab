#!/bin/sh
# CELL 2 — install the inference stack. Re-runnable: every step is a no-op
# when already satisfied, so run it again freely after a Studio restart.
#
# Why diffusers from git and not PyPI: QwenImage21Pipeline landed in PR #14804
# and postdates the 0.40 release — the official model card says
# "pip install git+https://github.com/huggingface/diffusers". A PyPI diffusers
# will import fine and then fail with ImportError on the pipeline class.
#
# Cost: $0 of GPU time beyond the install wall-clock (a few minutes of a
# session you are already paying for). Idempotent — no --force-reinstall, so
# this will not fight torch if the Studio already ships a working CUDA build.

set -e
cd "$HOME/qwenimage21" 2>/dev/null || { mkdir -p "$HOME/qwenimage21"; cd "$HOME/qwenimage21"; }

echo "=== 1/4 huggingface_hub + hf_transfer (multi-stream download) ==="
# hf_transfer is what makes 33.1 GB tolerable; without it the hub falls back
# to single-connection and the download crawls.
python3 -m pip install -q -U "huggingface_hub[hf_transfer]" hf_transfer

echo "=== 2/4 diffusers from git (QwenImage21Pipeline is not on PyPI yet) ==="
python3 -m pip install -q "git+https://github.com/huggingface/diffusers"

echo "=== 3/4 transformers >= 5.17 (the Qwen3-VL text encoder needs it) ==="
python3 -m pip install -q -U "transformers>=5.17" accelerate safetensors

echo "=== 4/4 serving deps (only needed for cell 6) ==="
python3 -m pip install -q fastapi "uvicorn[standard]" python-multipart pillow

echo ""
echo "=== verify the import chain actually resolves ==="
python3 - <<'PYEOF'
import sys
import torch, diffusers, transformers
print("torch      :", torch.__version__)
print("diffusers  :", diffusers.__version__)
print("transformers:", transformers.__version__)

# The single most important line in this cell. If the class is missing, the
# PyPI diffusers won the race above and we need the git one.
from diffusers import QwenImage21Pipeline
print("QwenImage21Pipeline: OK")

# And prove it is a bf16-class pipeline on this card, not an fp16 one. The
# T4 route had to quantize to INT8 precisely because this check fails there.
print("supported dtypes:", [d.__name__ for d in QwenImage21Pipeline.supported_inference_dtypes])
if torch.cuda.is_available():
    print("cuda:", torch.cuda.get_device_name(0))
PYEOF

echo ""
echo "SETUP_DONE — if the line above says QwenImage21Pipeline: OK you can go on"
