#!/bin/sh
# CELL 3b — status monitor for background download.
#
# Inspects downloaded model weights, vocoder files, and reference voice clips.

DIR="/kaggle/tmp/f5tts"
LOG="$DIR/logs/download.log"

echo "== download log tail =="
if [ -f "$LOG" ]; then
  tail -20 "$LOG"
else
  echo "(no download.log yet)"
fi

echo ""
echo "== model storage status =="
ls -lh "$DIR/models/F5TTS_v1_Base" 2>/dev/null || echo "F5-TTS model directory not ready yet"
ls -lh "$DIR/models/vocos-mel-24khz" 2>/dev/null || echo "Vocos directory not ready yet"

echo ""
echo "== reference voices status =="
ls -lh "$DIR/voices" 2>/dev/null || echo "Voices directory not ready yet"

echo ""
if grep -q "DOWNLOAD_OK" "$LOG" 2>/dev/null; then
  echo "DOWNLOAD_STATUS: COMPLETE (DOWNLOAD_OK present)"
else
  echo "DOWNLOAD_STATUS: IN_PROGRESS or NOT_STARTED"
fi
