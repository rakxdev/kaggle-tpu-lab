#!/bin/sh
# CELL 2 — Install IndexTTS-2.5 runtime via official uv lockfile and Cloudflare binary.
#
# Uses official 'uv sync' to create an isolated Python 3.11 environment from Bilibili's
# uv.lock, completely avoiding dependency conflicts with Kaggle's system Python.

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
echo "== 2. synchronizing official IndexTTS environment via uv =="
cd "$REPO"

# Check if uv is available
if ! command -v uv > /dev/null 2>&1; then
  pip install -q -U uv
fi

# Synchronize exact locked dependencies from uv.lock into .venv
echo "  running uv sync (isolated Python 3.11 environment)..."
uv sync --extra webui

echo "  adding API server stack to environment..."
uv pip install -q fastapi uvicorn soundfile pydub

VENV_PY="$REPO/.venv/bin/python"
echo "  isolated environment ready: $("$VENV_PY" --version)"

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
echo "== 4. verify IndexTTS import inside isolated environment =="
"$VENV_PY" -c "
import indextts
from indextts.infer_v2_5 import IndexTTS2
print('  IndexTTS2 engine imported successfully inside isolated environment ✓')
"

echo ""
echo "SETUP_OK"
