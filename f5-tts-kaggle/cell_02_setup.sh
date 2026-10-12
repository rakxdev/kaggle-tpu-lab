#!/bin/sh
# CELL 2 — install F5-TTS, vocoder, and API runtime dependencies.
#
# Sets up F5-TTS package, vocoder dependencies, Cloudflare tunnel binary,
# and verifies CUDA device visibility and library loading.

set -e

DIR="/kaggle/tmp/f5tts"
BIN="$DIR/bin"
mkdir -p "$BIN" "$DIR/models" "$DIR/voices" "$DIR/logs"

echo "== 1. check torch and torchaudio =="
python3 -c "
import torch
print('  PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
try:
    import torchaudio
    print('  TorchAudio:', torchaudio.__version__)
except Exception as e:
    print('  TorchAudio warning:', e)
"

echo ""
echo "== 2. install f5-tts & runtime stack =="
pip install -q --no-warn-script-location \
  f5-tts \
  fastapi \
  uvicorn \
  gradio \
  pydub \
  soundfile \
  pypinyin \
  rjieba \
  unidecode \
  cached_path

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
echo "== 4. verify F5-TTS import =="
python3 -c "
from f5_tts.api import F5TTS
print('  F5-TTS module loaded successfully')
"

echo ""
echo "SETUP_OK"
