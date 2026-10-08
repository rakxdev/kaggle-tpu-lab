#!/bin/sh
# CELL 0b — the kit clone / pull. Run SECOND, every session, before anything.
#
# This is always the same contract: the repo at /kaggle/working/kaggle-tpu-lab
# is a clean mirror of rakxdev/main. Any fix I push, you get with a re-run of
# this cell. `git clean -qfd` wipes untracked files INSIDE the repo dir only —
# models (in /kaggle/tmp/escha), logs and the heartbeat script outside it all
# survive, so re-running never costs a re-download.

cd /kaggle/working || exit 1

echo "== kit clone/pull =="
if [ -d kaggle-tpu-lab/.git ]; then
  git -C kaggle-tpu-lab fetch -q origin
  git -C kaggle-tpu-lab reset -q --hard origin/main
  git -C kaggle-tpu-lab clean -qfd
  echo "  pulled: $(git -C kaggle-tpu-lab log --oneline -1 | head -c 90)"
else
  git clone -q https://github.com/rakxdev/kaggle-tpu-lab || { echo "!! clone failed"; exit 1; }
  echo "  cloned: $(git -C kaggle-tpu-lab log --oneline -1 | head -c 90)"
fi

[ -d kaggle-tpu-lab/qwen38-escha-t4 ] && echo "  kit present: qwen38-escha-t4"
echo ""
echo "PULL_OK"