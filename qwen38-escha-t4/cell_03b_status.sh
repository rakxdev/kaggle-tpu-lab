#!/bin/sh
# CELL 3b — poll the download. Re-run until it prints DOWNLOAD_OK.
# Cost: $0. Also does the final byte check independently of the inner script.

WORK=/kaggle/tmp/escha
LOG="$WORK/download.log"

if [ ! -f "$LOG" ]; then echo "no download log — run cell_03_download.py first"; exit 1; fi
tail -4 "$LOG"

if grep -q DOWNLOAD_FAILED "$LOG"; then echo "== DOWNLOAD_FAILED =="; exit 1; fi
if ! grep -q DOWNLOAD_OK "$LOG";  then echo "== still downloading ==";          exit 0; fi

echo "== DOWNLOAD_OK — byte check =="
Q8E=10307703008; MTP=2926418048
G1=$(stat -c%s "$WORK/Escha-Qwen3.8-27B-W2-Q8E.gguf" 2>/dev/null || echo 0)
G2=$(stat -c%s "$WORK/Escha-Qwen3.8-27B-W2-MTP-F16-headQ4.gguf" 2>/dev/null || echo 0)
echo "main : $G1 / $Q8E"
echo "draft: $G2 / $MTP"
if [ "$G1" = "$Q8E" ] && [ "$G2" = "$MTP" ]; then
  echo "SIZES_OK"
else
  echo "!! size mismatch — re-run cell_03_download.py (hf_hub resumes)"
  exit 1
fi