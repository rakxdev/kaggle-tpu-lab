#!/bin/sh
# CELL 0b — Kit clone / clean-pull. Run SECOND in every Kaggle session.
#
# Keeps /kaggle/working/kaggle-tpu-lab synchronized with rakxdev/main.
# Untracked files inside the repo are cleaned, while external directories
# (/kaggle/tmp/indextts weights, logs, heartbeat) survive safely.

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

[ -d kaggle-tpu-lab/indextts-kaggle ] && echo "  kit ready: indextts-kaggle"
echo ""
echo "PULL_OK"
