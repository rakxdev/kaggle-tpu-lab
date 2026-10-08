#!/bin/sh
# CELL 2b — poll the build. Re-run until it prints BUILD_OK.
#
# Cost: $0. The build takes ~15-30 min on Kaggle's 4 vCPUs (sm_75-only cubins).
# If it prints BUILD_FAILED, grab the tail: it names the exact file that broke.

WORK=/kaggle/tmp/escha
LOG="$WORK/build.log"

if [ ! -f "$LOG" ]; then echo "no build log — run cell_02_setup.sh first"; exit 1; fi

if grep -q BUILD_OK "$LOG"; then
  echo "== BUILD_OK =="
  ls -la "$WORK"/llama.cpp-escha/build/bin/ | grep -E "llama-(server|cli|bench)"
  exit 0
fi
if grep -q BUILD_FAILED "$LOG"; then
  echo "== BUILD_FAILED — last 40 lines =="
  tail -40 "$LOG"
  exit 1
fi

echo "== still building — progress =="
grep -cE "^\[" "$LOG" 2>/dev/null | xargs -I{} echo "build steps done: {}"
tail -3 "$LOG"
echo ""
echo "(re-run this cell every few minutes)"