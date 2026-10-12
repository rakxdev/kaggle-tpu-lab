#!/bin/sh
# CELL 3b — Status monitor for background download.
#
# Inspects downloaded model weights and example voice/emotion clips.

DIR="/kaggle/tmp/indextts"
LOG="$DIR/logs/download.log"

echo "== download log tail =="
if [ -f "$LOG" ]; then
  tail -20 "$LOG"
else
  echo "(no download.log yet)"
fi

echo ""
echo "== checkpoints directory status =="
ls -lh "$DIR/checkpoints" 2>/dev/null || echo "Checkpoints directory not ready yet"

echo ""
echo "== example audio clips status =="
ls -lh "$DIR/examples" 2>/dev/null || echo "Examples directory not ready yet"

echo ""
if grep -q "DOWNLOAD_OK" "$LOG" 2>/dev/null; then
  echo "DOWNLOAD_STATUS: COMPLETE (DOWNLOAD_OK present)"
else
  echo "DOWNLOAD_STATUS: IN_PROGRESS or NOT_STARTED"
fi
