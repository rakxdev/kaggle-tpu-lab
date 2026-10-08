#!/bin/sh
# CELL 5 — bench: put real T4 numbers next to the author's 3090 numbers.
#
# Cost: ~5-10 min of quota. llama-bench measures the main model alone
# (speculative decoding is a server-level feature and llama-bench does not
# exercise it — the endpoint cell's proof measures MTP end-to-end).
#
# Reference (author, RTX 3090 @ 250 W, full offload):
#   pp512 700.4 tok/s · tg128 24.03 tok/s (Q8_0 head)
# The T4 has ~1/3 the memory bandwidth (320 vs 936 GB/s), so tg is expected
# around 7-10 tok/s; prefill uses the sm_75 tensor-core path. These runs turn
# those estimates into measurements.

WORK=/kaggle/tmp/escha
BIN="$WORK/llama.cpp-escha/build/bin/llama-bench"
MAIN="$WORK/Escha-Qwen3.8-27B-W2-Q8E.gguf"

[ -x "$BIN" ] || { echo "build first (cell_02b_status.sh)"; exit 1; }
[ -f "$MAIN" ] || { echo "model missing — run cell_03_download.py"; exit 1; }

# Separate kill/launch (HANDOFF rule).
pkill -f llama-bench 2>/dev/null
sleep 1

echo "== llama-bench: pp512 + tg128, full offload, fa on, q8 KV =="
# -fa 1 + q8 KV mirrors the server config; output is a markdown table.
"$BIN" -m "$MAIN" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -p 512 -n 128 -r 3 2>&1 | tee "$WORK/bench.log"

echo ""
echo "== VRAM after the run =="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo ""
echo "BENCH_OK — numbers are in $WORK/bench.log (copy them into the README)"