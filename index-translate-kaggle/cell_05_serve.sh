#!/bin/sh
# CELL 5 — the production endpoint: llama-server (with vision mmproj) + Cloudflare tunnel + watchdog.
#
# Exposes an OpenAI-compatible /v1/chat/completions endpoint that accepts:
#   1. Text translation (articles, paragraphs, subtitles)
#   2. Multimodal image translation (diagrams, comics, screenshots via base64 image_url)
#
# Hardware layout:
#   - GPU 0: 5.78 GB weights + 0.62 GB mmproj + 32k KV cache = ~7.5 GB used (leaves ~8.3 GB free).
#   - Zero PCIe split latency — runs at ~45-55 tok/s.
#   - Watchdog auto-stops server and tunnel at INDEX_MAX_MIN (default 480 min = 8 hours).

WORK=/kaggle/tmp/translate
MODELS_DIR="$WORK/models"
MAIN="$MODELS_DIR/Index-Translate-9B.Q4_K_M.gguf"
MMPROJ="$MODELS_DIR/Index-Translate-9B.mmproj-Q8_0.gguf"
PORT="${INDEX_PORT:-8000}"
CTX="${INDEX_CTX:-32768}"
MAX_MIN="${INDEX_MAX_MIN:-480}"
LOG="$WORK/server.log"
CFLOG="$WORK/cloudflared.log"

SRV="$WORK/llama-bin/llama-server"
[ -x "$SRV" ] || SRV="$WORK/llama.cpp/build/bin/llama-server"

[ -x "$SRV" ]   || { echo "!! llama-server binary missing — run cell_02_setup.sh"; exit 1; }
[ -f "$MAIN" ]  || { echo "!! text model missing — run cell_03_download.py"; exit 1; }
[ -f "$MMPROJ" ] || { echo "!! mmproj missing — run cell_03_download.py"; exit 1; }

# ---- 1. API Key -----------------------------------------------------------
if [ -z "$INDEX_API_KEY" ]; then
  INDEX_API_KEY="index-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  export INDEX_API_KEY
fi

# ---- 2. Watchdog ---------------------------------------------------------
WATCH="$WORK/watchdog.sh"
cat > "$WATCH" <<EOF
#!/bin/sh
LIMIT=\$(( ${MAX_MIN} * 60 )); T0=\$(date +%s)
while true; do
  NOW=\$(date +%s)
  if [ \$((NOW - T0)) -ge \$LIMIT ]; then
    pkill -f "llama-server.*${PORT}" 2>/dev/null
    pkill -f "[c]loudflared tunnel.*${PORT}" 2>/dev/null
    echo "\$(date) WATCHDOG: ${MAX_MIN} min limit reached — server and tunnel stopped" >> "$LOG"
    exit 0
  fi
  sleep 60
done
EOF
chmod +x "$WATCH"

# ---- 3. Relaunch cleanly (kill & launch separate — HANDOFF rule) ---------
pkill -f "index_watchdog_marker" 2>/dev/null
pkill -f "llama-server.*${PORT}" 2>/dev/null
pkill -f "[c]loudflared tunnel.*${PORT}" 2>/dev/null
sleep 2

echo "== launching llama-server (log: $LOG) =="
nohup "$SRV" \
  -m "$MAIN" \
  --mmproj "$MMPROJ" \
  -ngl 99 \
  --device CUDA0 \
  -c "$CTX" \
  -b 2048 \
  -ub 2048 \
  --port "$PORT" \
  --host 127.0.0.1 \
  --alias "Index-Translate-9B" \
  --api-key "$INDEX_API_KEY" \
  --chat-template-kwargs '{"enable_thinking":false}' \
  > "$LOG" 2>&1 &

nohup sh -c "exec -a index_watchdog_marker $WATCH" > /dev/null 2>&1 &
echo "server pid $! — watchdog armed (${MAX_MIN} min ceiling)"

echo "== waiting for model and vision encoder load =="
UP=0
for i in $(seq 1 45); do
  sleep 3
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" | grep -q '"ok"'; then
    UP=1
    break
  fi
done

if [ "$UP" != "1" ]; then
  echo "!! server failed to start within 135s — log tail:"
  tail -30 "$LOG"
  exit 1
fi

echo "✓ server healthy on port $PORT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo ""
echo "== connecting Cloudflare tunnel =="
if [ ! -x /tmp/cloudflared ]; then
  curl -sSL -o /tmp/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    || { echo "!! failed to download cloudflared"; exit 1; }
  chmod +x /tmp/cloudflared
fi

setsid nohup /tmp/cloudflared tunnel --url "http://127.0.0.1:${PORT}" \
  --no-autoupdate > "$CFLOG" 2>&1 < /dev/null &

URL=""
for i in $(seq 1 15); do
  sleep 4
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CFLOG" | head -1)
  [ -n "$URL" ] && break
done

if [ -z "$URL" ]; then
  echo "!! failed to acquire tunnel URL — log tail:"
  tail -15 "$CFLOG"
  exit 1
fi

echo ""
echo "############################################################"
echo "#  INDEX_ENDPOINT_OK"
echo "#"
echo "#  Public Base URL : $URL"
echo "#  API Endpoint    : $URL/v1/chat/completions"
echo "#  API Key         : $INDEX_API_KEY"
echo "#  Model Name      : Index-Translate-9B"
echo "#  Capabilities    : Text Translation + Multimodal Image Translation"
echo "#"
echo "#  =========================================================="
echo "#  Configuration for Immersive Translate (Firefox / Kiwi):"
echo "#  1. Translation Service: OpenAI (Custom)"
echo "#  2. Custom API interface address:"
echo "#     $URL/v1/chat/completions"
echo "#  3. APIKEY:"
echo "#     $INDEX_API_KEY"
echo "#  4. Model:"
echo "#     Index-Translate-9B"
echo "#  =========================================================="
echo "#"
echo "#  Kill command    : pkill -f llama-server; pkill -f cloudflared"
echo "############################################################"
