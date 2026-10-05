#!/bin/sh
# CELL 6b — expose the server publicly through ngrok, on the RESERVED domain.
#
# *** THIS CELL MAKES IT PUBLIC. Anyone who learns the API key can spend your
# *** credit through this URL. Only run it after cell 5's numbers convinced
# *** you, and remember the model license: Qwen Research — non-commercial.
#
# The authtoken is read from /tmp/ngrok_token (or $NGROK_TOKEN). It is NEVER
# hardcoded here and NEVER committed — this file lives on GitHub.
#   echo 'YOUR_TOKEN' > /tmp/ngrok_token && chmod 600 /tmp/ngrok_token
#
# The DOMAIN is your ngrok reserved domain (free accounts get one). Static
# across restarts, so the endpoint URL survives Studio restarts — no more
# new-URL-every-run. Note: a reserved domain binds to exactly ONE live tunnel;
# if the Strata lane endpoint is still up somewhere, shut it down first or
# ngrok will refuse this session with a "domain in use" error.
#
# Cost: the tunnel is free, but it keeps the GPU awake at ~$3.82/h. The cell 6
# watchdog still hard-stops everything at its deadline.

NGROK_TOKEN="${NGROK_TOKEN:-$(cat /tmp/ngrok_token 2>/dev/null)}"
DOMAIN="${NGROK_DOMAIN:-pseudoasymmetric-unbodied-sabine.ngrok-free.dev}"
PORT=8080
WORK="${QI21_WORK:-$HOME/qwenimage21}"
LOG="$WORK/ngrok.log"

case "$NGROK_TOKEN" in
  ""|PASTE_*) echo "no ngrok token: run  echo 'YOUR_TOKEN' > /tmp/ngrok_token  first"; exit 1 ;;
esac

# ---- 1. sanity: is the server actually up before we tunnel to it? ---------
# Tunnelling to a dead port produces a URL that 502s for everyone, which is
# confusing to debug later.
if ! curl -s --max-time 5 "http://127.0.0.1:$PORT/health" > /dev/null; then
  echo "!! nothing answering on 127.0.0.1:$PORT — run cell 06_serve.sh first"
  tail -20 "$WORK/serve.log" 2>/dev/null
  exit 1
fi
echo "local server is up"

# ---- 2. ngrok binary + auth ----------------------------------------------
if [ ! -x /tmp/ngrok ]; then
  curl -sSL -o /tmp/ngrok.tgz \
    https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz || exit 1
  tar xzf /tmp/ngrok.tgz -C /tmp || exit 1
  chmod +x /tmp/ngrok
fi
/tmp/ngrok config add-authtoken "$NGROK_TOKEN" > /dev/null 2>&1 \
  && echo "ngrok auth ok" || { echo "!! ngrok rejected the token"; exit 1; }

# ---- 3. launch the tunnel on the reserved domain --------------------------
# Kill before launch, as separate commands — the pkill pattern must not match
# this script's own argv, hence the [t] bracket trick.
pkill -f "ngrok [h]ttp" 2>/dev/null
sleep 2
DOMAIN_ARG=""
[ -n "$DOMAIN" ] && DOMAIN_ARG="--domain=$DOMAIN"
nohup /tmp/ngrok http $DOMAIN_ARG --log=stdout --log-level=info "$PORT" > "$LOG" 2>&1 &

sleep 12

URL=$(grep -oE 'https://[a-z0-9-]+\.(ngrok-free\.app|ngrok\.io|ngrok-free\.dev)' "$LOG" | head -1)

echo ""
if [ -z "$URL" ]; then
  echo "!! no public URL in the log — paste the log below:"
  tail -20 "$LOG"
  exit 1
fi

echo "############################################################"
echo "#  PUBLIC ENDPOINT LIVE"
echo "#  URL     : $URL   (reserved domain — stable across restarts)"
echo "#  API KEY : (the one cell 6 printed)"
echo "#"
echo "#  test it:"
echo "#    curl -s $URL/health"
echo "#    curl -s $URL/v1/images/generations -H 'Authorization: Bearer <KEY>' \\"
echo "#      -H 'Content-Type: application/json' \\"
echo "#      -d '{\"prompt\":\"a red panda barista\",\"width\":512,\"height\":512,\"steps\":20}' \\"
echo "#      | python3 -c 'import sys,json,base64;d=json.load(sys.stdin);print(d[\"seconds\"],\"s\");open(\"o.png\",\"wb\").write(base64.b64decode(d[\"data\"][0][\"b64_json\"]))'"
echo ""
echo "#  the response is OpenAI /v1/images/generations shaped, so an OpenAI"
echo "#  client works unmodified:"
echo "#    from openai import OpenAI"
echo "#    OpenAI(base_url=\"$URL/v1\", api_key=\"<KEY>\").images.generate("
echo "#        model=\"qwen-image-2.1\", prompt=\"a red panda barista\")"
echo ""
echo "#  STOP BILLING NOW:  pkill -f qi21_[s]erver.py ; pkill -f \"ngrok [h]ttp\""
echo "#  watchdog still caps this at the cell 6 limit."
echo "############################################################"
