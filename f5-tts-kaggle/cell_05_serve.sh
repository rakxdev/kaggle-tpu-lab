#!/bin/sh
# CELL 5 — Launch F5-TTS FastAPI server + Web Studio + Cloudflare quick tunnel.
#
# Hardware: Kaggle 2x T4 (CUDA:0 allocated).
# Exposes:
# 1. Interactive Web Studio: https://*.trycloudflare.com/
# 2. OpenAI-compatible API : https://*.trycloudflare.com/v1/audio/speech
# 3. Health & Voices check : https://*.trycloudflare.com/health & /v1/audio/voices
#
# Watchdog kills the server after MAX_MIN (default 240 = 4 hours) to guard GPU hours.

PORT=8000
MAX_MIN=240
DIR="/kaggle/tmp/f5tts"
LOG="$DIR/logs/server.log"
CFLOG="$DIR/logs/cloudflared.log"
CF="$DIR/bin/cloudflared"

mkdir -p "$DIR/logs"

# Ensure cloudflared binary is available
if [ ! -x "$CF" ]; then
  echo "== downloading cloudflared =="
  curl -sSL -o "$CF" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CF"
fi

# Watchdog definition
WATCH="sleep $((MAX_MIN * 60)); pkill -f 'server.py.*${PORT}'; pkill -f '[c]loudflared.*${PORT}'"

echo "== stopping any prior server / tunnel on port $PORT =="
pkill -f "f5_watchdog_marker" 2>/dev/null
pkill -f "server.py.*${PORT}" 2>/dev/null
pkill -f "[c]loudflared tunnel.*${PORT}" 2>/dev/null
sleep 2

echo "== launching F5-TTS FastAPI server (log: $LOG) =="
nohup python3 /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/server.py \
  --port "$PORT" \
  --host 127.0.0.1 \
  ${F5_API_KEY:+--api-key "$F5_API_KEY"} \
  > "$LOG" 2>&1 &

nohup sh -c "exec -a f5_watchdog_marker $WATCH" > /dev/null 2>&1 &
echo "server pid $! — watchdog armed (${MAX_MIN} min ceiling)"

echo "== waiting for F5-TTS model load and healthcheck =="
UP=0
for i in $(seq 1 45); do
  sleep 2
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" | grep -q '"ok"'; then
    UP=1
    break
  fi
done

if [ "$UP" != "1" ]; then
  echo "!! server failed to respond within 90s — log tail:"
  tail -30 "$LOG"
  exit 1
fi

echo "✓ server healthy on port $PORT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo ""
echo "== launching Cloudflare tunnel =="
setsid nohup "$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$CFLOG" 2>&1 < /dev/null &

echo "== waiting for public URL =="
URL=""
for i in $(seq 1 30); do
  sleep 2
  URL=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" "$CFLOG" 2>/dev/null | head -1)
  [ -n "$URL" ] && break
done

if [ -z "$URL" ]; then
  echo "!! failed to acquire tunnel URL — log tail:"
  tail -15 "$CFLOG"
  exit 1
fi

echo ""
echo "############################################################"
echo "#  F5TTS_ENDPOINT_OK"
echo "#"
echo "#  Web Studio UI   : $URL"
echo "#  Speech API      : $URL/v1/audio/speech"
echo "#  Voices Listing  : $URL/v1/audio/voices"
echo "#  Health Check    : $URL/health"
echo "#"
echo "#  Model           : F5-TTS v1 Base (~1.5 GB weights, ~2.8 GB VRAM)"
echo "#  Hardware        : Kaggle T4 (CUDA:0)"
echo "#  API Format      : OpenAI Compatible (/v1/audio/speech)"
echo "#"
echo "#  cURL Test Example:"
echo "#  curl -X POST $URL/v1/audio/speech \\"
echo "#    -H 'Content-Type: application/json' \\"
echo "#    -d '{\"model\":\"f5-tts\",\"input\":\"Hello world\",\"voice\":\"nature_en\"}' \\"
echo "#    --output output.wav"
echo "#"
echo "#  Kill command    : pkill -f server.py; pkill -f cloudflared"
echo "############################################################"
