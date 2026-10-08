#!/usr/bin/env python3
# CELL 4 — smoke test: one real generation with the full serving flag set.
#
# Cost: ~2-3 min of quota (model load + one short generation). This is the
# go/no-go proof that the W2 kernel actually RUNS on sm_75 — compiling is not
# running. It also prints per-GPU VRAM after load, which fills in the one
# number the README cannot know in advance: the GDN (linear-attention)
# recurrent state size.
#
# Success sentinel: ESCHA_SMOKE_OK. Anything else — paste the whole output.

import json
import os
import subprocess
import sys
import time
import urllib.request

WORK = "/kaggle/tmp/escha"
MAIN = f"{WORK}/Escha-Qwen3.8-27B-W2-Q8E.gguf"
DRAFT = f"{WORK}/Escha-Qwen3.8-27B-W2-MTP-F16-headQ4.gguf"
PORT = 8099  # smoke port; the endpoint cell uses 8080
CTX = 8192   # small on purpose — smoke is about correctness, not context

for p in (MAIN, DRAFT):
    if not os.path.exists(p):
        sys.exit(f"missing {p} — run cell_03_download.py first")
SRV = f"{WORK}/llama.cpp-escha/build/bin/llama-server"
if not os.path.exists(SRV):
    sys.exit("llama-server not built — run cell_02b_status.sh until BUILD_OK")

print("== devices ==")
os.system(f"{WORK}/llama.cpp-escha/build/bin/llama-cli --list-devices 2>/dev/null | head -5")

# Kill and launch are separate commands (HANDOFF rule).
subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
time.sleep(3)

# The exact flag set the author documents for MTP serving, with the draft
# pinned to the second T4 so GPU0 keeps all its headroom for KV + state.
CMD = [
    SRV, "-m", MAIN, "-md", DRAFT,
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
    "--device", "CUDA0", "-ngl", "999",
    "--spec-draft-device", "CUDA1", "--spec-draft-ngl", "999",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
    "-c", str(CTX), "-np", "1", "-b", "2048", "-ub", "2048",
    "--temp", "0", "--top-k", "1", "--jinja",
    "--host", "127.0.0.1", "--port", str(PORT),
]
print("== launching (log: smoke.log) ==")
log = open(f"{WORK}/smoke.log", "w")
proc = subprocess.Popen(CMD, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)

def get(path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.load(r)

up = False
for _ in range(90):  # the 9.6 GB load takes 1-3 min
    time.sleep(4)
    try:
        h = get("/health")
        if h.get("status") == "ok":
            up = True
            break
    except Exception:
        pass
    if proc.poll() is not None:
        log.close()
        print(open(f"{WORK}/smoke.log").read()[-3000:])
        sys.exit("!! llama-server died during load — log above")

if not up:
    proc.terminate()
    print(open(f"{WORK}/smoke.log").read()[-3000:])
    sys.exit("!! server never became healthy — log above")

print("== loaded — VRAM per GPU ==")
os.system("nvidia-smi --query-gpu=index,memory.used,memory.total "
          "--format=csv,noheader")

print("== generate ==")
body = json.dumps({
    "messages": [{"role": "user", "content": "What is the capital of France? "
                                              "Answer in one word."}],
    "max_tokens": 32, "temperature": 0,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                             data=body, method="POST")
req.add_header("Content-Type", "application/json")
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.load(r)
dt = time.time() - t0
text = out["choices"][0]["message"]["content"].strip()
ntok = out["usage"]["completion_tokens"]
print(f"answer : {text!r}")
print(f"{ntok} tokens in {dt:.1f}s = {ntok / dt:.2f} tok/s (incl. load-free warmup)")
print(f"timings: {json.dumps(out.get('timings', {}))}")

# Leave the working setup visible in the log, then stop the smoke server so
# the bench/serve cells start clean.
proc.terminate()
print("ESCHA_SMOKE_OK" if "paris" in text.lower() else "ESCHA_SMOKE_WRONG_ANSWER")