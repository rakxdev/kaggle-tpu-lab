# Phase 0 / cell 0 — ntfy command relay, v3 (429-hardened + account auth)
#
# v1 died on ntfy's anonymous rate limit: the idle loop polled ~1x/second and
# one 429 on publish crashed the agent (seen live). v2 fixed the mechanics.
# v3 adds the account token: a Kaggle VM's shared egress IP is rate-limited by
# ntfy as anonymous even when WE behave — authenticated requests lift the cap
# (this fixed the GPU session's "results never arrive" wall, seen live).
#   - receives commands on a BLOCKING stream (no polling at all)
#   - publishes with retry/backoff honoring Retry-After; failed publishes
#     wait in a pending queue instead of crashing
#   - persists seen-ids + since to /kaggle/working/agent_state.json, so a
#     restarted agent never re-executes old commands
#   - Bearer token on BOTH publish and the command stream
# Pure stdlib, port 443 only. FRESH TOPIC PAIR PER SESSION — a second agent on
# the same topics executes another session's commands (seen live).

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

CMD_TOPIC = "ktl-cmd-6d59c49709ad"   # TPU session 2026-09-20 (fresh pair)
OUT_TOPIC = "ktl-out-61a425e8277f"
NTFY_TOKEN = "PASTE_YOUR_NTFY_TOKEN"  # the same ntfy account token used before
STATE = "/kaggle/working/agent_state.json"
CMD_TIMEOUT_S = 900
OUT_TRUNC = 3000


def hdr():
    h = {"Content-Type": "application/json"}
    if NTFY_TOKEN and not NTFY_TOKEN.startswith("PASTE_"):
        h["Authorization"] = f"Bearer {NTFY_TOKEN}"
    return h


def post(obj):
    data = json.dumps(obj).encode()
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(f"https://ntfy.sh/{OUT_TOPIC}",
                                           data=data, headers=hdr()),
                    timeout=60) as r:
                r.read()
            return True
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After") or min(5 * 2 ** attempt, 120))
            print(f"[post {e.code} — waiting {wait}s]", flush=True)
            time.sleep(wait)
            last = e
        except Exception as e:
            print(f"[post error — retrying]", repr(e)[:120], flush=True)
            time.sleep(5)
            last = e
    print("[post failed permanently]", repr(last), flush=True)
    return False


def run(cmd):
    try:
        r = subprocess.run(cmd, shell=True, text=True, capture_output=True,
                           timeout=CMD_TIMEOUT_S)
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return {"rc": r.returncode, "out": out[-OUT_TRUNC:]}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": f"TIMEOUT after {CMD_TIMEOUT_S}s (partial work may "
                                  "still be running; use a nohup+log pattern next)"}
    except Exception as e:
        return {"rc": -1, "out": repr(e)[:OUT_TRUNC]}


seen, since = set(), int(time.time()) - 10
if os.path.exists(STATE):
    try:
        st = json.load(open(STATE))
        seen = set(st.get("seen", []))
        since = max(since, int(st.get("since", 0)))
    except Exception:
        pass


def save():
    try:
        json.dump({"seen": sorted(seen), "since": since}, open(STATE, "w"))
    except Exception:
        pass


if post({"agent": "up-v3-auth", "session": "tpu-flashnext-20260920"}):
    print(f"agent online v3 (authenticated)  cmds: ntfy.sh/{CMD_TOPIC}", flush=True)

pending = []


def flush():
    for r in list(pending):
        if post(r):
            pending.remove(r)


while True:
    flush()
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{CMD_TOPIC}/json?since={since}", headers=hdr())
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:  # ntfy streams lines, keep-alives keep it warm
                try:
                    ev = json.loads(raw.decode())
                except Exception:
                    continue
                if ev.get("event") != "message":
                    continue
                since = max(since, ev["time"])
                save()
                try:
                    msg = json.loads(ev.get("message", "{}"))
                except Exception:
                    continue
                i = msg.get("id")
                if i is None or i in seen:
                    continue
                seen.add(i)
                save()
                cmd = msg.get("cmd", "true")
                print(f"[exec id {i}] {cmd[:120]}", flush=True)
                pending.append({"id": i, **run(cmd)})
                flush()
    except Exception as e:
        print("[stream reset]", repr(e)[:160], flush=True)
        time.sleep(5)
