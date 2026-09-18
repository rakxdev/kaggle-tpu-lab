# Phase 0 / cell 0 — ntfy command relay (debugging vehicle for CPU sessions)
#
# Turns the notebook session into a remotely driven shell: polls a private
# ntfy topic for {"id": N, "cmd": "..."} messages, executes each once via
# /bin/sh, and publishes {"id": N, "rc": .., "out": ..} to a second topic.
# Pure stdlib, port 443 only, no accounts. The topic strings are the only
# credential: keep them private, they die with the session.
#
# This cell blocks (it IS the agent). Everything else goes through the relay.

import json
import subprocess
import time
import urllib.request

CMD_TOPIC = "ktl-cmd-41cbd7210e5b"
OUT_TOPIC = "ktl-out-f28de123fc89"
CMD_TIMEOUT_S = 900
OUT_TRUNC = 3000


def ntfy_get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()


def post(obj):
    req = urllib.request.Request(f"https://ntfy.sh/{OUT_TOPIC}",
                                 data=json.dumps(obj).encode())
    urllib.request.urlopen(req, timeout=30).read()


def run(cmd):
    try:
        r = subprocess.run(cmd, shell=True, text=True, capture_output=True,
                           timeout=CMD_TIMEOUT_S)
        out = r.stdout or ""
        if r.stderr:
            out += "\n[stderr]\n" + r.stderr
        return {"rc": r.returncode, "out": out[-OUT_TRUNC:]}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": f"TIMEOUT after {CMD_TIMEOUT_S}s (partial work may "
                                   "still be running; use a nohup+log pattern next)"}
    except Exception as e:
        return {"rc": -1, "out": repr(e)[:OUT_TRUNC]}


since = int(time.time()) - 10
seen = set()
post({"agent": "up",
      "machine": subprocess.run("hostname; free -g | head -2; df -h /kaggle/working | tail -1",
                                shell=True, text=True, capture_output=True).stdout})
print(f"agent online  cmds: ntfy.sh/{CMD_TOPIC}  out: ntfy.sh/{OUT_TOPIC}", flush=True)

while True:
    try:
        body = ntfy_get(f"https://ntfy.sh/{CMD_TOPIC}/json?poll=1&since={since}")
    except Exception:
        time.sleep(3)
        continue
    for line in body.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "message":
            continue
        since = max(since, ev["time"])
        try:
            msg = json.loads(ev.get("message", "{}"))
        except Exception:
            continue
        i = msg.get("id")
        if i is None or i in seen:
            continue
        seen.add(i)
        cmd = msg.get("cmd", "true")
        print(f"[exec id {i}] {cmd[:120]}", flush=True)
        post({"id": i, **run(cmd)})
