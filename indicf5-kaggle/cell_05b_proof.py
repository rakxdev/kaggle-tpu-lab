#!/usr/bin/env python3
"""CELL 5b — Automated proof & verification of live IndicF5 server.

Validates the running HTTP endpoint:
1. Checks /health and /v1/audio/voices metadata.
2. Synthesizes a test narration using cloned Shubh voice via /v1/audio/speech.
3. Inspects WAV headers (24,000 Hz, mono PCM), byte count, and RTF metrics.
4. Saves verification audio to /kaggle/tmp/indicf5/proof_shubh.wav.
5. Emits sentinel PROOF_OK.
"""

import sys
import time
import json
import wave
import io
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
BASE_URL = BASE_URL.rstrip("/")
OUT_WAV = Path("/kaggle/tmp/indicf5/proof_shubh.wav")
OUT_WAV.parent.mkdir(parents=True, exist_ok=True)

print(f"== 1. Checking Server Health ({BASE_URL}/health) ==")
req_health = urllib.request.Request(f"{BASE_URL}/health")
try:
    with urllib.request.urlopen(req_health, timeout=10) as resp:
        health_data = json.loads(resp.read().decode("utf-8"))
        print(f"  Status       : {health_data.get('status')}")
        print(f"  Engine       : {health_data.get('engine')}")
        print(f"  Device       : {health_data.get('device')}")
        print(f"  Default Voice: {health_data.get('default_voice')}")
        print(f"  VRAM Alloc   : {health_data.get('vram_allocated_gb')} GiB")
        assert health_data.get("status") == "ok", "Server health status is not ok"
except urllib.error.URLError as e:
    print(f"!! Failed to connect to server: {e}")
    sys.exit(1)

print(f"\n== 2. Checking Voice Catalog ({BASE_URL}/v1/audio/voices) ==")
req_voices = urllib.request.Request(f"{BASE_URL}/v1/audio/voices")
with urllib.request.urlopen(req_voices, timeout=10) as resp:
    voices_data = json.loads(resp.read().decode("utf-8"))
    voice_ids = [v["id"] for v in voices_data.get("voices", [])]
    print(f"  Available voices: {voice_ids}")
    assert "shubh" in voice_ids, "'shubh' not found in voice catalog"

print(f"\n== 3. Testing Speech Synthesis with Cloned Shubh Voice ==")
payload = {
    "model": "indicf5",
    "input": "बारिश की एक ठंडी शाम थी, और शहर में चारों तरफ सन्नाटा था।",
    "voice": "shubh",
    "speed": 1.0,
    "normalize": True
}
payload_bytes = json.dumps(payload).encode("utf-8")
req_speech = urllib.request.Request(
    f"{BASE_URL}/v1/audio/speech",
    data=payload_bytes,
    headers={"Content-Type": "application/json"},
    method="POST"
)

t0 = time.time()
with urllib.request.urlopen(req_speech, timeout=60) as resp:
    audio_bytes = resp.read()
    latency = time.time() - t0
    dur_header = resp.headers.get("X-Audio-Duration", "--")
    norm_text = resp.headers.get("X-Normalized-Text", "")

print(f"  HTTP Status    : {resp.status} (OK)")
print(f"  Latency        : {latency:.2f}s")
print(f"  Audio Received : {len(audio_bytes):,} bytes")

# Validate WAV file structure
with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
    channels = wav_file.getnchannels()
    sample_rate = wav_file.getframerate()
    frames = wav_file.getnframes()
    duration = frames / float(sample_rate)

print(f"  Audio Channels : {channels} (Mono)")
print(f"  Sample Rate    : {sample_rate} Hz (High-Fidelity)")
print(f"  Audio Duration : {duration:.2f}s")
rtf = latency / duration if duration > 0 else 0.0
print(f"  Real-Time Factor: {rtf:.3f}x")

assert channels == 1, f"Expected 1 channel, got {channels}"
assert sample_rate == 24000, f"Expected 24000 Hz, got {sample_rate}"
assert duration > 1.0, f"Audio too short ({duration:.2f}s)"
assert len(audio_bytes) > 20000, "Audio payload too small"

OUT_WAV.write_bytes(audio_bytes)
print(f"  Saved Proof WAV: {OUT_WAV} ({OUT_WAV.stat().st_size / 1024:.1f} KB)")

print("\nPROOF_OK")
