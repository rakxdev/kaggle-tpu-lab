#!/bin/sh
# CELL 3b — Lightweight status monitor for download progress & disk usage.

DIR="/kaggle/tmp/indicf5"
CHECKPOINTS="$DIR/checkpoints"

echo "== IndicF5 Download & Disk Status =="
if [ -d "$CHECKPOINTS" ]; then
  echo "Checkpoints directory size:"
  du -sh "$CHECKPOINTS"
  ls -lh "$CHECKPOINTS" | head -15
else
  echo "Checkpoints directory not created yet."
fi

echo ""
echo "Shubh reference voice:"
ls -lh "$DIR"/shubh_ref.* 2>/dev/null || echo "  (shubh_ref not staged yet)"

echo ""
echo "Filesystem usage:"
df -h /kaggle/tmp /kaggle/working 2>/dev/null
