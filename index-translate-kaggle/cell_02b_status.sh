#!/bin/sh
# CELL 2b — poll build progress. Re-run until it prints BUILD_OK.

WORK=/kaggle/tmp/translate
LOG="$WORK/build.log"

for i in 1 2 3 4 5; do
  [ -f "$LOG" ] && break
  sleep 1
done

if [ ! -f "$LOG" ]; then
  echo "no build log yet at $LOG — run cell_02_setup.sh first"
  exit 1
fi

if grep -q BUILD_OK "$LOG"; then
  echo "== BUILD_OK =="
  ls -lh "$WORK/llama-bin/llama-server" 2>/dev/null || ls -lh "$WORK/llama.cpp/build/bin/llama-server" 2>/dev/null
  exit 0
fi

if grep -q BUILD_FAILED "$LOG"; then
  echo "== BUILD_FAILED — last 30 lines =="
  tail -30 "$LOG"
  exit 1
fi

echo "== still compiling — progress =="
grep -cE "^\[" "$LOG" 2>/dev/null | xargs -I{} echo "build steps done: {}"
tail -3 "$LOG"
echo ""
echo "(re-run this cell every few minutes)"
