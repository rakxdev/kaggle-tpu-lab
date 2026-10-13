#!/bin/sh
# CELL 2 — Environment & dependency setup (100% Pure uv & wheels).
#
# Hardware target: Kaggle 2x T4 (SM75, 15.83 GiB VRAM each).
# Provisions an isolated Python 3.11 virtualenv at /kaggle/tmp/indicf5/venv.
# Uses uv for ultra-fast, pre-compiled wheel installations:
# - PyTorch 2.5.1 + CUDA 12.4
# - AI4Bharat IndicF5 (f5_tts engine)
# - indic-transliteration (pure Python, zero C build requirements)
# - FastAPI, Uvicorn, SoundFile, PyDub, Transformers
# - Applies the Mode A character-count duration patch.

set -e

DIR="/kaggle/tmp/indicf5"
VENV="$DIR/venv"
BIN="$DIR/bin"
CHECKPOINTS="$DIR/checkpoints"
LOGS="$DIR/logs"
mkdir -p "$BIN" "$CHECKPOINTS" "$LOGS"

echo "== 1. isolated Python 3.11 environment via uv =="
command -v uv > /dev/null 2>&1 || pip install -q -U uv

if [ ! -f "$VENV/bin/python" ]; then
  echo "  provisioning Python 3.11 standalone runtime..."
  uv venv --python 3.11 "$VENV"
else
  echo "  found existing venv at $VENV"
fi

PY="$VENV/bin/python"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

echo "  using: $($PY --version) at $PY"

echo ""
echo "== 2. PyTorch & CUDA 12.4 installation (uv) =="
uv pip install --python "$PY" -q "torch>=2.0,<2.6" "torchaudio>=2.0,<2.6" "numpy<2.1" --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "== 3. IndicF5 & Audio processing stack (uv) =="
uv pip install --python "$PY" -q git+https://github.com/AI4Bharat/IndicF5.git
uv pip install --python "$PY" -q fastapi uvicorn soundfile pydub huggingface_hub hf_transfer "transformers<5.0,>=4.40" indic-transliteration matplotlib-inline "pyarrow<19"

echo ""
echo "== 4. cloudflared tunnel client =="
if [ ! -x "$BIN/cloudflared" ]; then
  echo "  fetching cloudflared-linux-amd64..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "$BIN/cloudflared"
  chmod +x "$BIN/cloudflared"
fi
echo "  cloudflared: $($BIN/cloudflared --version | head -c 50)"

echo ""
echo "== 5. Duration patch (Mode A fix) =="
$PY -c "
import glob, sys
from pathlib import Path

candidates = glob.glob('$VENV/**/f5_tts/infer/utils_infer.py', recursive=True)
if not candidates:
    print('  WARNING: utils_infer.py not located yet.')
    sys.exit(0)

patched_count = 0
for p in candidates:
    src = Path(p).read_text(encoding='utf-8')
    if 'sum(1 for c in ref_text if not c.isspace())' in src:
        print(f'  already patched: {p}')
        patched_count += 1
        continue
    
    if 'len(ref_text.encode(\"utf-8\"))' in src:
        new_src = src.replace('len(ref_text.encode(\"utf-8\"))', 'sum(1 for c in ref_text if not c.isspace())')
        new_src = new_src.replace('len(gen_text.encode(\"utf-8\"))', 'sum(1 for c in gen_text if not c.isspace())')
        Path(p).write_text(new_src, encoding='utf-8')
        print(f'  successfully patched: {p}')
        patched_count += 1
    else:
        print(f'  no target pattern in: {p}')

print(f'  duration patch verified on {patched_count} location(s).')
"

echo ""
echo "== 6. Environment verification =="
$PY -c "
import sys, torch, transformers, f5_tts, soundfile
print('  Python version :', sys.version.split()[0])
print('  PyTorch version:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('  Transformers   :', transformers.__version__)
print('  F5-TTS module  :', f5_tts.__file__)
"

echo ""
echo "SETUP_OK"
