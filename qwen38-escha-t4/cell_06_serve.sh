#!/bin/sh
# CELL 6 — the public endpoint: llama-server (MTP draft on GPU1) + cloudflared
# quick tunnel + watchdog. THIS CELL IS THE PRODUCT.
#
# *** PUBLIC ENDPOINT: anyone with the URL + key uses your Kaggle GPU quota.
# *** The 30 h/week allowance is the budget; the watchdog below stops server
# *** AND tunnel at ESCHA_MAX_MIN (default 480 = 8 h of the 12 h session cap)
# *** so a forgotten tab cannot silently drain the week.
#
# Launch flags are the author's documented MTP set, with one Kaggle-specific
# change: the 2.73 GB draft is pinned to CUDA1 (--spec-draft-device) so GPU0
# keeps the full KV + GDN-state headroom. Reading order for the numbers:
#   GPU0  main weights 9.6 GiB + compute + KV(q8: 32 KiB/token) + GDN state
#   GPU1  draft 2.73 GiB
#
# MTP contract (from the model card — the two settings that quietly kill the
# speedup if you change them):
#   -np 1                more slots split the verify batch; MTP gain vanishes
#   --temp 0 --top-k 1   greedy; at temp 1 acceptance drops ~3.1 -> ~2.2 tokens
# Single stream, ~10-14 expected T4 tok/s. Concurrent users QUEUE (the server
# has one slot) — that is the trade for MTP; see README for the multi-user flag set.

WORK=/kaggle/tmp/escha
BIN="$WORK/llama.cpp-escha/build/bin/llama-server"
MAIN="$WORK/Escha-Qwen3.8-27B-W2-Q8E.gguf"
DRAFT="$WORK/Escha-Qwen3.8-27B-W2-MTP-F16-headQ4.gguf"
PORT="${ESCHA_PORT:-8080}"
CTX="${ESCHA_CTX:-32768}"
MAX_MIN="${ESCHA_MAX_MIN:-480}"
LOG="$WORK/server.log"
CFLOG="$WORK/cloudflared.log"

[ -x "$BIN" ]   || { echo "no llama-server — finish cell_02b"; exit 1; }
[ -f "$MAIN" ]  || { echo "no model — run cell_03_download.py"; exit 1; }
[ -f "$DRAFT" ] || { echo "no draft — run cell_03_download.py"; exit 1; }

# ---- key ------------------------------------------------------------------
if [ -z "$ESCHA_API_KEY" ]; then
  ESCHA_API_KEY="escha-$(head -c 18 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  export ESCHA_API_KEY
fi

# ---- watchdog (separate process; kills server + tunnel at the deadline) ---
WATCH="$WORK/watchdog.sh"
cat > "$WATCH" <<EOF
#!/bin/sh
LIMIT=\$(( ${MAX_MIN} * 60 )); T0=\$(date +%s)
while true; do
  NOW=\$(date +%s)
  if [ \$((NOW - T0)) -ge \$LIMIT ]; then
    pkill -f llama-server 2>/dev/null
    pkill -f "[c]loudflared tunnel" 2>/dev/null
    echo "\$(date) WATCHDOG: ${MAX_MIN} min reached — stopped" >> "$LOG"
    exit 0
  fi
  sleep 60
done
EOF
chmod +x "$WATCH"

# ---- relaunch clean (kill and launch are SEPARATE — HANDOFF rule) ---------
pkill -f "qi21_watchdog_marker\|escha_watchdog_marker" 2>/dev/null
pkill -f llama-server 2>/dev/null
pkill -f "[c]loudflared tunnel" 2>/dev/null
sleep 3

echo "== launching llama-server (log $LOG) =="
nohup "$BIN" \
  -m "$MAIN" -md "$DRAFT" \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --device CUDA0 -ngl 999 \
  --spec-draft-device CUDA1 --spec-draft-ngl 999 \
  -fa on -ctk q8_0 -ctv q8_0 \
  -c "$CTX" -np 1 -b 2048 -ub 2048 \
  --temp 0 --top-k 1 --jinja \
  --api-key "$ESCHA_API_KEY" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
echo "server pid $!"

nohup sh -c "exec -a escha_watchdog_marker $WATCH" > /dev/null 2>&1 &
echo "watchdog armed — hard stop at ${MAX_MIN} min"

echo "== waiting for load (9.6 GB to VRAM + draft: 1-3 min) =="
UP=0
for i in $(seq 1 60); do
  sleep 5
  if curl -s --max-time 3 "http://127.0.0.1:${PORT}/health" | grep -q '"ok"'; then
    UP=1; break
  fi
done
if [ "$UP" != "1" ]; then
  echo "!! server not healthy after 5 min — log tail:"
  tail -30 "$LOG"; exit 1
fi
echo "server healthy:"
curl -s "http://127.0.0.1:${PORT}/health"; echo
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo ""
echo "== self-test (1 short completion) =="
curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Authorization: Bearer $ESCHA_API_KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' \
  | head -c 400; echo

echo ""
echo "== cloudflared quick tunnel =="
if [ ! -x /tmp/cloudflared ]; then
  curl -sSL -o /tmp/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    || { echo "!! cloudflared download failed"; exit 1; }
  chmod +x /tmp/cloudflared
fi
setsid nohup /tmp/cloudflared tunnel --url "http://127.0.0.1:${PORT}" \
  --no-autoupdate > "$CFLOG" 2>&1 < /dev/null &
URL=""
for i in $(seq 1 12); do
  sleep 5
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CFLOG" | head -1)
  [ -n "$URL" ] && break
done
if [ -z "$URL" ]; then
  echo "!! no tunnel URL — tail:"; tail -10 "$CFLOG"; exit 1
fi
curl -s --max-time 20 -o /dev/null -w "tunnel probe: http=%{http_code}\n" "$URL/health" \
  || echo "(probe failed — tunnel may need a few more seconds)"

echo ""
echo "############################################################"
echo "#  ESCHA_ENDPOINT_OK"
echo "#  URL     : $URL"
echo "#  docs    : $URL (OpenAI-compatible /v1/chat/completions)"
echo "#  API KEY : $ESCHA_API_KEY"
echo "#  ctx     : $CTX (q8 KV — raise ESCHA_CTX and re-run for more)"
echo "#  MTP     : on (draft on CUDA1, greedy, single slot)"
echo "#  kill all: pkill -f llama-server; pkill -f cloudflared"
echo "############################################################"
echo "#  then run cell_06b_proof.py from OUTSIDE (your laptop):"
echo "#    python3 cell_06b_proof.py $URL \$KEY"