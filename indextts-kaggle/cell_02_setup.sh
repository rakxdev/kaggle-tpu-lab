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

echo "  installing indextts package (editable)..."
pip install -q -e "$REPO" --no-deps

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
import indextts
from indextts.infer_v2_5 import IndexTTS2
print('  IndexTTS2 engine imported successfully')
"

echo ""
echo "SETUP_OK"
