#!/bin/sh
# CELL 3b — poll download status and verify byte-exact model sizes.

WORK=/kaggle/tmp/translate
LOG="$WORK/download.log"
MODELS_DIR="$WORK/models"

if [ ! -f "$LOG" ]; then
  echo "no download log — run cell_03_download.py first"
  exit 1
fi

tail -4 "$LOG"

if grep -q DOWNLOAD_FAILED "$LOG"; then
  echo "== DOWNLOAD_FAILED =="
  exit 1
fi

if ! grep -q DOWNLOAD_OK "$LOG"; then
  echo "== still downloading =="
  exit 0
fi

echo "== DOWNLOAD_OK — performing independent size verification =="
M_SIZE=5780090304
V_SIZE=620690272

G_M=$(stat -c%s "$MODELS_DIR/Index-Translate-9B.Q4_K_M.gguf" 2>/dev/null || echo 0)
G_V=$(stat -c%s "$MODELS_DIR/Index-Translate-9B.mmproj-Q8_0.gguf" 2>/dev/null || echo 0)

echo "text model : $G_M / $M_SIZE bytes"
echo "vision mmproj: $G_V / $V_SIZE bytes"

if [ "$G_M" = "$M_SIZE" ] && [ "$G_V" = "$V_SIZE" ]; then
  echo "SIZES_OK — both model files verified byte-exact!"
  exit 0
else
  echo "!! size mismatch — re-run cell_03_download.py to resume"
  exit 1
fi
