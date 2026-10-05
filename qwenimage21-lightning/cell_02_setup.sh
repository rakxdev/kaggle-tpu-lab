#!/bin/sh
# CELL 2 — install the inference stack. Re-runnable: every step is a no-op
# when already satisfied, so run it again freely after a Studio restart.
#
# WHY A VENV, and why diffusers from git:
#  - The base Lightning Studio's system Python is PEP 668 externally-managed
#    (Ubuntu 24.04): pip refuses to touch it ("externally-managed-environment").
#    There is no conda on the image either, but uv IS preinstalled at
#    /usr/local/bin/uv — so this cell builds a venv with uv and installs into
#    that. The image also does NOT ship torch (cell 1 proves it), so torch is
#    installed here too.
#  - QwenImage21Pipeline landed in PR #14804 and postdates the 0.40 release —
#    the model card says "pip install git+https://github.com/huggingface/diffusers".
#    A PyPI diffusers imports fine and then dies on ImportError for the class.
#
# Cost: $0 of GPU time — this is CPU/network work inside a session you are
# already paying for (the download step is the slow part, ~2-6 GB of wheels).
#
# AFTER THIS CELL, run every other cell with the venv python on PATH:
#   export PATH="$HOME/qwenimage21/venv/bin:$PATH"

set -e
WORK="$HOME/qwenimage21"
mkdir -p "$WORK"
cd "$WORK"

# ---- 0. venv via uv (instant) --------------------------------------------
if [ ! -x "$WORK/venv/bin/python" ]; then
  echo "=== 0/5 creating venv with uv ==="
  if command -v uv >/dev/null 2>&1; then
    uv venv "$WORK/venv" --python "$(command -v python3)"
  else
    python3 -m venv "$WORK/venv"
  fi
else
  echo "=== 0/5 venv exists, skipping ==="
fi
PY="$WORK/venv/bin/python"
UVPIP() { command -v uv >/dev/null 2>&1 && uv pip install --python "$PY" "$@" || "$PY" -m pip install "$@"; }

echo "=== 1/5 torch (PyPI default wheel bundles CUDA-12 runtime; driver 580 OK) ==="
UVPIP torch

echo "=== 2/5 huggingface_hub + hf_transfer (multi-stream download) ==="
UVPIP "huggingface_hub[hf_transfer]" hf_transfer

echo "=== 3/5 diffusers from git (QwenImage21Pipeline is not on PyPI yet) ==="
UVPIP "git+https://github.com/huggingface/diffusers"

echo "=== 4/5 transformers >= 5.17 (the Qwen3-VL text encoder needs it) ==="
UVPIP "transformers>=5.17" accelerate safetensors peft  # peft: LoRA loading (fast lane)

echo "=== 5/5 torchvision + serving deps — BOTH load-bearing, not optional ==="
# torchvision: the checkpoint ships a Qwen3VL *video* processor class, and
# transformers raises ImportError for Qwen3VLVideoProcessor without it — even
# for pure text-to-image where no video is ever processed. Seen live 2026-10-05.
UVPIP torchvision
# python3.12-dev: Triton JIT-compiles its CUDA driver wrapper with gcc against
# Python.h on first CUDA launch. Without the dev headers every run dies at
# "fatal error: Python.h: No such file or directory" — also seen live. libcuda
# itself is present; only the headers are missing. Needs sudo; Studios allow it.
if [ ! -f /usr/include/python3.12/Python.h ] && command -v sudo >/dev/null 2>&1; then
  sudo -n apt-get install -y -q python3.12-dev || echo "!! could not install python3.12-dev — triton will fail later; paste this log"
fi
UVPIP fastapi "uvicorn[standard]" python-multipart pillow

echo ""
echo "=== verify the import chain actually resolves ==="
"$PY" - <<'PYEOF'
import torch, diffusers, transformers
print("torch      :", torch.__version__)
print("diffusers  :", diffusers.__version__)
print("transformers:", transformers.__version__)
print("cuda avail :", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# The single most important line in this cell. If the class is missing, the
# PyPI diffusers won the race above and we need the git one.
from diffusers import QwenImage21Pipeline
print("QwenImage21Pipeline: OK")
# NOTE: older docs/discussions mention `supported_inference_dtypes`; current
# diffusers main (0.41.0.dev0) does NOT expose it on this class — verified live
# 2026-10-05 on H200. Do not assert it unconditionally.
print("supported dtypes:", getattr(QwenImage21Pipeline, "supported_inference_dtypes",
                                  "attribute not exposed on this diffusers build"))

# The dtype question this kit cares about. H200 (sm_90) does bf16 natively;
# fp16 is NOT offered by this pipeline and would NaN anyway.
a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul finite:", bool(torch.isfinite(a @ a).all()))
PYEOF

echo ""
echo "export PATH=\"$WORK/venv/bin:\$PATH\"   # <- prefix every later cell with this"
echo "SETUP_DONE — if the line above says QwenImage21Pipeline: OK you can go on"
