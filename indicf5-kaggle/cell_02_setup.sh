#!/bin/sh
# CELL 2 — Environment & dependency setup.
#
# Hardware target: Kaggle 2x T4 (SM75, 15.83 GiB VRAM each).
# Provisions an isolated Python 3.11 virtualenv at /kaggle/tmp/indicf5/venv,
# completely bypassing Kaggle's system Python 3.13 site-packages conflicts.
# Installs fairseq, ai4bharat-transliteration, IndicXlit models, IndicF5,
# applies the Mode A character-count duration patch, and fetches cloudflared.

set -e

DIR="/kaggle/tmp/indicf5"
VENV="$DIR/venv"
BIN="$DIR/bin"
CHECKPOINTS="$DIR/checkpoints"
LOGS="$DIR/logs"
mkdir -p "$BIN" "$CHECKPOINTS" "$LOGS"

echo "== 1. isolated Python 3.11 environment via uv =="
command -v uv > /dev/null 2>&1 || pip install -q -U uv

if [ ! -f "$VENV/bin/pip" ]; then
  echo "  provisioning Python 3.11 standalone runtime with pip..."
  rm -rf "$VENV"
  uv venv --seed --python 3.11 "$VENV"
else
  echo "  found existing venv at $VENV"
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "  using: $($PY --version) at $PY"

echo ""
echo "== 2. PyTorch & CUDA 12.4 installation =="
uv pip --python "$PY" install -q "torch>=2.0,<2.6" "torchaudio>=2.0,<2.6" "numpy<2.1" --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "== 3. fairseq & ai4bharat-transliteration build =="
uv pip --python "$PY" install -q "setuptools<69" "Cython<3" wheel

_FSDIR=$(mktemp -d)
echo "  fetching fairseq v0.12.2 source..."
curl -fsSL "https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.12.2.tar.gz" -o "$_FSDIR/fairseq.tar.gz"
tar xzf "$_FSDIR/fairseq.tar.gz" -C "$_FSDIR"
printf "0.12.2\n" > "$_FSDIR/fairseq-0.12.2/fairseq/version.txt"

echo "  compiling fairseq C extensions (isolated venv)..."
"$PIP" install "$_FSDIR/fairseq-0.12.2/" --no-build-isolation -q
rm -rf "$_FSDIR"

echo "  installing ai4bharat-transliteration..."
uv pip --python "$PY" install -q "ai4bharat-transliteration==1.1.3"
uv pip --python "$PY" install -q --upgrade setuptools

echo ""
echo "== 4. IndicF5 & Studio dependencies =="
uv pip --python "$PY" install -q git+https://github.com/AI4Bharat/IndicF5.git
uv pip --python "$PY" install -q fastapi uvicorn soundfile pydub "huggingface_hub<2.0,>=1.16.0" hf_transfer "transformers<5.0,>=4.40"

echo ""
echo "== 5. IndicXlit weights pre-cache =="
_XLIT_PKG_DIR=$($PY -c "import ai4bharat.transliteration, os; print(os.path.dirname(ai4bharat.transliteration.__file__))" 2>/dev/null || echo "")
if [ -n "$_XLIT_PKG_DIR" ]; then
  _XLIT_MODELS_DIR="$_XLIT_PKG_DIR/transformer/models/en2indic/v1.0"
  mkdir -p "$_XLIT_MODELS_DIR"
  if [ ! -f "$_XLIT_MODELS_DIR/transformer/indicxlit.pt" ]; then
    echo "  caching IndicXlit en→indic model (121 MB)..."
    curl -fsSL -o /tmp/indicxlit-model.zip "https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/indicxlit-en-indic-v1.0.zip"
    unzip -q -o /tmp/indicxlit-model.zip -d "$_XLIT_MODELS_DIR"
    rm -f /tmp/indicxlit-model.zip
  fi
  if [ ! -d "$_XLIT_MODELS_DIR/word_prob_dicts" ]; then
    echo "  caching IndicXlit word prob dicts (163 MB)..."
    curl -fsSL -o /tmp/indicxlit-dicts.zip "https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/word_prob_dicts.zip"
    unzip -q -o /tmp/indicxlit-dicts.zip -d "$_XLIT_MODELS_DIR"
    rm -f /tmp/indicxlit-dicts.zip
  fi
  echo "  IndicXlit pre-cache ready."
fi

echo ""
echo "== 6. cloudflared tunnel client =="
if [ ! -x "$BIN/cloudflared" ]; then
  echo "  fetching cloudflared-linux-amd64..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "$BIN/cloudflared"
  chmod +x "$BIN/cloudflared"
fi
echo "  cloudflared: $($BIN/cloudflared --version | head -c 50)"

echo ""
echo "== 7. Duration patch (Mode A fix) =="
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
echo "== 8. Environment verification =="
$PY -c "
import torch, transformers, f5_tts, soundfile
print('  Python version :', sys.version.split()[0])
print('  PyTorch version:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('  Transformers   :', transformers.__version__)
print('  F5-TTS module  :', f5_tts.__file__)
"

echo ""
echo "SETUP_OK"
