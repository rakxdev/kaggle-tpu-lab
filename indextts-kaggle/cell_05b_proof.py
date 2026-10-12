#!/usr/bin/env python3
"""
CELL 5b — Automated proof client for IndexTTS-2.5 server and emotion API.

Verifies:
1. Health endpoint: /health
2. Speech API with 8-axis emotion vector: POST /v1/audio/speech
3. Audio format inspection: validates RIFF/WAVE header and byte count
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path

DIR = Path("/kaggle/tmp/indextts")
CFLOG = DIR / "logs" / "cloudflared.log"
PORT = 8000
API_KEY = os.environ.get("INDEX_API_KEY", "")

# Determine base URL (prefer Cloudflare public URL if available, else localhost)
base_url = f"http://127.0.0.1:{PORT}"
if CFLOG.exists():
    with open(CFLOG, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    import re
    match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", content)
    if match:
        base_url = match.group(0)

print(f"== 1. probing IndexTTS-2.5 endpoint at {base_url} ==")

def call_api(path, data=None, method="GET"):
    url = f"{base_url}{path}"
    headers = {"User-Agent": "IndexTTS-Proof/1.0"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=120)

# Check health
try:
    with call_api("/health") as resp:
        hdata = json.load(resp)
        print("  /health response :", hdata)
        assert hdata.get("status") == "ok", "Server health status was not ok"
except Exception as e:
    print(f"!! healthcheck failed: {e}", file=sys.stderr)
    sys.exit(1)

# Check speech synthesis with emotion vector
print("\n== 2. testing emotional speech synthesis (/v1/audio/speech) ==")
t0 = time.time()
test_text = "Incredible! This is high-fidelity neural speech with rich emotion on NVIDIA T4."
# Happy 0.9, Surprised 0.5
emo_vector = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]

try:
    with call_api("/v1/audio/speech", {
        "model": "indextts-2.5",
        "input": test_text,
        "lang": "EN",
        "emo_vector": emo_vector,
        "emo_alpha": 0.85,
    }, method="POST") as resp:
        audio_bytes = resp.read()
    dt = time.time() - t0
    print(f"  Input text       : {test_text!r}")
    print(f"  Emotion vector   : {emo_vector}")
    print(f"  Response status  : {resp.status}")
    print(f"  Audio byte count : {len(audio_bytes):,} bytes (received in {dt:.2f}s)")

    # Validate WAV header
    assert len(audio_bytes) > 10000, "Audio response too small to be valid WAV"
    assert audio_bytes[:4] == b"RIFF", "Audio file missing RIFF header"
    assert audio_bytes[8:12] == b"WAVE", "Audio file missing WAVE format marker"
    print("  Header check     : Valid RIFF/WAVE header verified ✓")

    out_file = DIR / "proof_out.wav"
    with open(out_file, "wb") as f:
        f.write(audio_bytes)
    print(f"  Saved test audio : {out_file}")

except Exception as e:
    print(f"!! emotional speech synthesis test failed: {e}", file=sys.stderr)
    sys.exit(1)

print("\nPROOF_OK")
