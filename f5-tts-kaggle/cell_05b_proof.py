#!/usr/bin/env python3
"""
CELL 5b — Automated verification client for F5-TTS server and speech API.

Verifies:
1. Health endpoint: /health
2. Preset voices listing: /v1/audio/voices
3. Speech synthesis: POST /v1/audio/speech
4. Audio format inspection: validates RIFF/WAVE header and minimum byte count
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path

DIR = Path("/kaggle/tmp/f5tts")
CFLOG = DIR / "logs" / "cloudflared.log"
PORT = 8000
API_KEY = os.environ.get("F5_API_KEY", "")

# Determine base URL (prefer Cloudflare public URL if available, else localhost)
base_url = f"http://127.0.0.1:{PORT}"
if CFLOG.exists():
    with open(CFLOG, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    import re
    match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", content)
    if match:
        base_url = match.group(0)

print(f"== 1. probing F5-TTS endpoint at {base_url} ==")

def call_api(path, data=None, method="GET"):
    url = f"{base_url}{path}"
    headers = {"User-Agent": "F5TTS-Proof/1.0"}
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

# Check voices
try:
    with call_api("/v1/audio/voices") as resp:
        vdata = json.load(resp)
        voices = vdata.get("voices", {})
        print(f"  /v1/audio/voices : {len(voices)} voices found ({', '.join(voices.keys())})")
except Exception as e:
    print(f"!! voices list failed: {e}", file=sys.stderr)
    sys.exit(1)

# Check speech synthesis
print("\n== 2. testing speech synthesis (/v1/audio/speech) ==")
t0 = time.time()
test_text = "Testing neural audio synthesis via F5-TTS on Kaggle T4."
try:
    with call_api("/v1/audio/speech", {
        "model": "f5-tts",
        "input": test_text,
        "voice": "nature_en",
        "speed": 1.0,
    }, method="POST") as resp:
        audio_bytes = resp.read()
    dt = time.time() - t0
    print(f"  Input text       : {test_text!r}")
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
    print(f"!! speech synthesis test failed: {e}", file=sys.stderr)
    sys.exit(1)

print("\nPROOF_OK")
