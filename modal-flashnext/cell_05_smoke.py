# CELL 5 — smoke test through the public URL. Run right after cell 4c.
# First call cold-starts the A100 (~2-4 min: volume read + weights into VRAM)
# and BILLS that whole wait; after that it stays warm for scaledown_window.

import json
import os
import time
import urllib.request

URL = "PASTE-THE-DEPLOY-URL-HERE"  # from cell 4c output, e.g. https://...modal.run
KEY = os.environ["APP_API_KEY"]    # set in cell 4a — same notebook session


def call(path, payload=None, timeout=1500):
    req = urllib.request.Request(
        URL.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.time() - t0


print("/health:", f"{call('/health')[1]:.1f}s (cold start — billed)")
models, _ = call("/v1/models")
print("models:", [m["id"] for m in models["data"]])

body, dt = call("/v1/chat/completions", {
    "model": "flashnext-iq3s",
    "messages": [{"role": "user", "content": "Reply with exactly: ENDPOINT OK"}],
    "max_tokens": 20,
})
print("reply:", repr(body["choices"][0]["message"]["content"]), f"({dt:.1f}s incl. startup)")

body, dt = call("/v1/chat/completions", {
    "model": "flashnext-iq3s",
    "messages": [{"role": "user", "content": "Write 200 words on the history of computing."}],
    "max_tokens": 256,
})
u = body["usage"]
print(f"warm decode: {u['completion_tokens']} tok in {dt:.1f}s = {u['completion_tokens'] / dt:.1f} tok/s client-side")
print("SMOKE_DONE")
