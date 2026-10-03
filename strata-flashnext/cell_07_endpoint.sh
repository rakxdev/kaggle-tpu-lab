#!/bin/sh
# CELL 7 — the public endpoint, GLM-kernel style: static ngrok domain (stable
# across sessions — no more new-URL-per-run), API key, both APIs (OpenAI /v1 +
# Anthropic /v1/messages), and a keepalive watchdog so a forgotten session
# shuts itself down before burning quota (the GLM design's keepalive_min).
KEY=$(cat /kaggle/working/strata_api_key 2>/dev/null)
if [ -z "$KEY" ]; then
  KEY="strata-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  echo "$KEY" > /kaggle/working/strata_api_key
fi
DOMAIN="pseudoasymmetric-unbodied-sabine.ngrok-free.dev"

# 1. API key + this domain in Strata's config (it 403s unknown Host names)
python3 - "$KEY" "$DOMAIN" <<'EOF'
import json, sys, os
key, domain = sys.argv[1], sys.argv[2]
p = os.path.expanduser("~/.config/strata/settings.json")
try: cfg = json.load(open(p))
except Exception: cfg = {}
cfg["api_key"] = key
hosts = cfg.get("allowed_hosts") or []
if domain not in hosts: hosts.append(domain)
cfg["allowed_hosts"] = hosts
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(cfg, open(p, "w"), indent=1)
print("config -> api_key + allowed_hosts:", host)
EOF

# 2. keepalive watchdog (GLM design): shuts server + tunnel down after 480 min
cat > /kaggle/working/keepalive_watchdog.sh <<'EOF'
#!/bin/sh
LIMIT=$((480 * 60)); T0=$(date +%s)
while true; do
  NOW=$(date +%s)
  if [ $((NOW - T0)) -ge $LIMIT ]; then
    pkill -f "port 8080"; pkill -f "ngrok http"; pkill -f "ngrok.*$1"
    echo "$(date) keepalive reached — server + tunnel stopped" >> /kaggle/working/keepalive.log
    exit 0
  fi
  sleep 120
done
EOF
chmod +x /kaggle/working/keepalive_watchdog.sh
pkill -f keepalive_watchdog 2>/dev/null
nohup /kaggle/working/keepalive_watchdog.sh "$DOMAIN" > /dev/null 2>&1 &

# 3. server (re)start + ngrok tunnel on the static domain
pkill -f "port 8080" 2>/dev/null; sleep 3
pkill -f "ngrok http" 2>/dev/null; sleep 1
cd /kaggle/working/Strata || exit 1
STRATA_ALLOWED_HOSTS="$DOMAIN" nohup ./setup.sh --port 8080 > /kaggle/working/strata_serve.log 2>&1 &
nohup /tmp/ngrok http --domain="$DOMAIN" 8080 > /kaggle/working/ngrok.log 2>&1 &
sleep 25

echo ""
echo "############################################################"
echo "#  READY — the endpoint is live"
echo "#  ENDPOINT : https://$DOMAIN   (OpenAI /v1 + Anthropic /v1/messages)"
echo "#  API KEY  : $KEY"
echo "#  MODEL    : qwen3.8-flash-next-coder (IQ1_M), 2x T4"
echo "#"
echo "#  OpenAI-compatible clients (Codex CLI, aider, opencode, ...):"
echo "#    export OPENAI_BASE_URL=https://$DOMAIN/v1"
echo "#    export OPENAI_API_KEY=$KEY"
echo "#"
echo "#  Claude Code (Anthropic-compatible, like the GLM kernel):"
echo "#    export ANTHROPIC_BASE_URL=https://$DOMAIN"
echo "#    export ANTHROPIC_AUTH_TOKEN=$KEY"
echo "#    export ANTHROPIC_MODEL=qwen3.8-flash-next-coder"
echo "#    export ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-flash-next-coder"
echo "#"
echo "#  curl check (the skip-header bypasses ngrok's browser warning):"
echo "#    curl https://$DOMAIN/v1/models -H \"Authorization: Bearer $KEY\" \\"
echo "#         -H 'ngrok-skip-browser-warning: true'"
echo "#"
echo "#  Keepalive: the server shuts down after 480 min on its own."
echo "############################################################"
