#!/bin/sh
# CELL 2b — poll the build. Re-run until it prints BUILD_OK.
#
# Cost: $0. The build takes ~15-30 min on Kaggle's 4 vCPUs (sm_75-only cubins).
# If it prints BUILD_FAILED, grab the tail: it names the exact file that broke.

WORK=/kaggle/tmp/escha
LOG="$WORK/build.log"

# Give the log file a few seconds to appear if cell_02b is clicked right after cell_02
for i in 1 2 3 4 5; do
  [ -f "$LOG" ] && break
  sleep 1
done

if [ ! -f "$LOG" ]; then
  echo "no build log yet at $LOG"
  echo "running build processes: $(pgrep -f 'do_build\|cmake' || echo 'none')"
  exit 1
fi

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