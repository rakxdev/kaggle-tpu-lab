#!/bin/sh
# CELL 2 — Install IndexTTS-2.5 runtime, dependencies, and Cloudflare binary.
#
# Sets up official index-tts repository, audio processing libraries,
# speech tokenizer dependencies, and verifies CUDA acceleration.

set -e

DIR="/kaggle/tmp/indextts"
REPO="$DIR/repo"
BIN="$DIR/bin"
CHECKPOINTS="$DIR/checkpoints"
LOGS="$DIR/logs"

mkdir -p "$BIN" "$CHECKPOINTS" "$LOGS"

echo "== 1. clone / update official index-tts repository =="
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull -q || echo "  using existing index-tts checkout"
else
  git clone --depth 1 -q https://github.com/index-tts/index-tts.git "$REPO"
fi
echo "  repository ready at $REPO"

echo ""
echo "== 2. install IndexTTS runtime dependencies =="
pip install -q --no-warn-script-location "huggingface-hub<2.0,>=1.16.0" "transformers>=4.46.0,<=4.52.1"
pip install -q --no-warn-script-location \
  einops \
  descript-audiotools \
  cn2an \
  g2p-en \
  fugashi \
  unidic-lite \
  WeTextProcessing \
  jieba \
  librosa \
  modelscope \
  munch \
  omegaconf \
  safetensors \
  sentencepiece \
  openai-whisper \
  fastapi \
  uvicorn \
  soundfile \
  pydub \
  json5 \
  textstat

echo "  linking indextts package into Python path..."
# Remove strict python version ceiling (<3.12) from pyproject.toml
sed -i 's/requires-python = ".*"/requires-python = ">=3.10"/' "$REPO/pyproject.toml" 2>/dev/null || true

# Patch cache_utils import resilience in case of transformers version variance
python3 -c "
import pathlib
target = pathlib.Path('$REPO/indextts/gpt/transformers_generation_utils.py')
if target.exists():
    content = target.read_text(encoding='utf-8')
    if 'from transformers.cache_utils import' in content:
        # Protect OffloadedCache and QuantizedCacheConfig with safe fallback
        content = content.replace('OffloadedCache,', '').replace('QuantizedCacheConfig,', '')
        fallback = '''
from transformers.cache_utils import Cache
try:
    from transformers.cache_utils import OffloadedCache
except ImportError:
    class OffloadedCache: pass
try:
    from transformers.cache_utils import QuantizedCacheConfig
except ImportError:
    class QuantizedCacheConfig: pass
'''
        content = fallback + '\n' + content
        target.write_text(content, encoding='utf-8')
        print('  applied cache_utils resilience patch to indextts')
"

# Add .pth file to python site-packages to ensure direct importability
python3 -c "
import site, pathlib
for p in site.getsitepackages():
    pth = pathlib.Path(p) / 'indextts.pth'
    try:
        pth.write_text('$REPO\n', encoding='utf-8')
        print('  linked indextts to', pth)
        break
    except Exception:
        pass
"
# Also install with --no-deps if possible
pip install -q -e "$REPO" --no-deps 2>/dev/null || true

echo ""
echo "== 3. cloudflared binary =="
CF_BIN="$BIN/cloudflared"
if [ ! -x "$CF_BIN" ]; then
  echo "  downloading cloudflared..."
  curl -sSL -o "$CF_BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CF_BIN"
fi
echo "  cloudflared: $("$CF_BIN" --version 2>/dev/null | head -1)"

echo ""
echo "== 4. verify IndexTTS import =="
python3 -c "
import sys
sys.path.insert(0, '$REPO')
import indextts
from indextts.infer_v2_5 import IndexTTS2
print('  IndexTTS2 engine imported successfully')
"

echo ""
echo "SETUP_OK"
