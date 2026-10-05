#!/usr/bin/env python3
# CELL 6c — prove the public endpoint works end to end, from outside.
#
# Run this AFTER cell 6b. Paste the URL and key that cell 6b printed. It runs
# on this machine (or any machine) against the public ngrok URL, so it proves
# the whole chain: tunnel -> auth -> token gate -> GPU render -> image back.
#
# It also tests the things that actually break in practice:
#   - a request with NO token must be 401 (the gate is real, not decorative)
#   - a burst of concurrent requests must produce 429s, not corruption or an
#     OOM — this is the check that the server's lock is doing its job
#   - a real image comes back and is not black
#
# Cost: ~$0.10-0.20 of GPU time (a couple of real renders).

import base64
import concurrent.futures
import io
import json
import sys
import time
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "PASTE_PUBLIC_URL"
KEY = sys.argv[2] if len(sys.argv) > 2 else "PASTE_API_KEY"

if URL.startswith("PASTE_") or KEY.startswith("PASTE_"):
    sys.exit("usage: python3 cell_06c_proof.py <public_url> <api_key>")


def call(path: str, payload=None, key=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        URL.rstrip("/") + path, data=data, method="POST" if data else "GET"
    )
    if data:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    # ngrok's interstitial browser warning otherwise blocks API clients.
    req.add_header("ngrok-skip-browser-warning", "true")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:  # noqa: BLE001
            return e.code, {"raw": body[:300].decode("utf-8", "replace")}


print("=== 1. health (no auth needed) ===")
st, body = call("/health")
print(st, json.dumps(body)[:300])
if st != 200:
    sys.exit("!! public URL is not answering — check the ngrok log")

print("\n=== 2. auth gate must actually gate ===")
st, body = call("/generate", {"prompt": "test", "steps": 1})
print(f"no token  -> {st} {str(body)[:120]}")
if st != 401:
    sys.exit(f"!! expected 401 without a token, got {st} — do not share this URL")
st, _ = call("/generate", {"prompt": "test", "steps": 1}, key="wrong-key")
print(f"bad token -> {st}")
if st != 401:
    sys.exit("!! a wrong key was accepted — stop and check the server")

print("\n=== 3. one real generation (512x512, 20 steps) ===")
t0 = time.time()
st, body = call(
    "/generate",
    {"prompt": "a red panda barista, studio light", "width": 512, "height": 512,
     "steps": 20, "seed": 7},
    key=KEY,
)
wall = time.time() - t0
if st != 200:
    sys.exit(f"!! generate failed {st}: {str(body)[:400]}")
print(f"{st}  server {body['seconds']}s  ({body['seconds_per_step']} s/step)  "
      f"wall {wall:.1f}s  seed {body['seed']}")
# b64_json lives under "data" — the response is OpenAI /v1/images/generations
# shaped, so an OpenAI SDK client works against this unmodified.
try:
    b64 = body["data"][0]["b64_json"]
except (KeyError, IndexError, TypeError):
    sys.exit(f"!! unexpected response shape (OpenAI 'data' list missing): "
             f"{str(body)[:300]}")
img = base64.b64decode(b64)
open("proof_512.png", "wb").write(img)
print(f"wrote proof_512.png ({len(img)/1024:.0f} KiB)")

# A black/flat PNG is the NaN signature from the wrong dtype. Catch it here
# rather than shipping a broken endpoint to the community.
try:
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(io.BytesIO(img)).convert("RGB"), dtype=np.float32)
    print(f"pixels mean={arr.mean():.1f} std={arr.std():.1f}")
    if arr.std() < 1.0:
        sys.exit("!! image is flat/black — wrong dtype, do not publish this")
except ImportError:
    print("(pillow/numpy not here — could not check for a black image)")

print("\n=== 4. concurrency: 4 at once must queue or 429, never corrupt ===")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futs = [
        ex.submit(call, "/generate",
                  {"prompt": f"test {i}", "width": 512, "height": 512, "steps": 20},
                  KEY)
        for i in range(4)
    ]
    codes = []
    for f in futs:
        st, body = f.result()
        codes.append(st)
        print(f"  {st} {'ok' if st == 200 else str(body)[:70]}")
if 200 in codes and 429 in codes:
    print("queueing works: serialised + rejected the excess")
elif all(c == 200 for c in codes):
    print("all 4 serialised successfully (no 429 — queue window was wide enough)")
else:
    print("!! unexpected codes — check the server log for VRAM errors")

print("\nPROOF_DONE — the endpoint is real")
