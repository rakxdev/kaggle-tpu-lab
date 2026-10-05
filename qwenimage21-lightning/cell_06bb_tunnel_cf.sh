#!/bin/sh
# CELL 6bb — expose the server publicly through a Cloudflare quick tunnel.
#
# *** THIS CELL MAKES IT PUBLIC. Anyone who learns the API key can spend your
# *** credit through this URL. Only run it after cell 5's numbers convinced
# *** you, and remember the model license: Qwen Research — non-commercial.
#
# WHY CLOUDFLARE AND NOT NGROK: ngrok's free plan meters bandwidth (1 GB/mo).
# A single 2K PNG is ~9-12 MB, so a handful of concurrent users exhausts the
# month in minutes and every visitor then sees ERR_NGROK_725 "Network
# bandwidth exceeded" — the endpoint looks broken through no fault of ours.
# Cloudflare quick tunnels impose no such cap and need no account.
#
# TRADE-OFF: a quick tunnel's hostname is EPHEMERAL — it changes on every
# restart (something-random.trycloudflare.com). That is fine for a test
# endpoint: the docs page reads its own origin at runtime, so every code
# example and copy button re-points itself to whatever URL you are serving
# from. Send people the new URL after a restart; nothing else needs editing.
#
# Cost: the tunnel is free, but it keeps the GPU awake at ~$3.82/h. The cell 6
# watchdog still hard-stops everything at its deadline.

PORT="${QI21_PORT:-8080}"
WORK="${QI21_WORK:-$HOME/qwenimage21}"
LOG="$WORK/cf.log"
BIN=/tmp/cloudflared

# ---- 1. sanity: is the server actually up before we tunnel to it? ---------
if ! curl -s --max-time 5 "http://127.0.0.1:$PORT/health" > /dev/null; then
  echo "!! nothing answering on 127.0.0.1:$PORT — run cell 06_serve.sh first"
  exit 1
fi

# ---- 2. the binary (self-healing: /tmp is wiped on Studio restart) --------
if [ ! -x "$BIN" ]; then
  echo "fetching cloudflared..."
  curl -sSL -o "$BIN" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    || { echo "!! download failed"; exit 1; }
  chmod +x "$BIN"
fi
"$BIN" --version || exit 1

# ---- 3. relaunch cleanly --------------------------------------------------
# Kill and launch are SEPARATE commands (HANDOFF rule): a combined pkill -f
# can match its own launch line and kill the new tunnel.
pkill -f "[c]loudflared tunnel" 2>/dev/null
sleep 2

cd "$WORK" || exit 1
setsid nohup "$BIN" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
  > "$LOG" 2>&1 < /dev/null &
echo "tunnel launching — log $LOG"

# ---- 4. read the assigned URL --------------------------------------------
# The quick tunnel prints its hostname into the log within ~10-20 s.
sleep 20
URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1)

if [ -z "$URL" ]; then
  echo "!! no URL yet — check the log:"
  tail -15 "$LOG"
  exit 1
fi

echo ""
echo "############################################################"
echo "#  PUBLIC URL:  $URL"
echo "#"
echo "#  docs page  : $URL/            (open in a browser)"
echo "#  health      : $URL/health"
echo "#"
echo "#  Send this URL + your API key to whoever you want to"
echo "#  share with. The docs page rewrites its own examples, so"
echo "#  nothing else needs changing after a restart."
echo "#"
echo "#  kill tunnel : pkill -f cloudflared"
echo "############################################################"

# ---- 5. prove it answers --------------------------------------------------
sleep 2
echo ""
echo "=== reachable? ==="
curl -s --max-time 15 -o /dev/null -w "docs http=%{http_code}\n" "$URL/" || echo "(not answering yet — give it a few seconds)"