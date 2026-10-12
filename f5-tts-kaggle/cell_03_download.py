#!/usr/bin/env python3
"""
CELL 3 — Download F5-TTS v1 Base weights, vocoder, and studio reference voices.

Downloads:
1. SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors (~1.5 GB) + vocab.txt
2. charactr/vocos-mel-24khz config and weights (~50 MB)
3. Bundled 5-10s clean reference voice clips + transcripts into /kaggle/tmp/f5tts/voices/
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path

# Enable high-speed Rust-based transfer if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("!! huggingface_hub missing, run cell 0 / 2 first", file=sys.stderr)
    sys.exit(1)

BASE_DIR = Path("/kaggle/tmp/f5tts")
MODEL_DIR = BASE_DIR / "models"
VOICES_DIR = BASE_DIR / "voices"
LOG_DIR = BASE_DIR / "logs"

for p in (MODEL_DIR, VOICES_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

print("== 1. downloading F5-TTS v1 Base weights ==")
t0 = time.time()
f5_model_file = hf_hub_download(
    repo_id="SWivid/F5-TTS",
    filename="F5TTS_v1_Base/model_1250000.safetensors",
    local_dir=str(MODEL_DIR),
)
f5_vocab_file = hf_hub_download(
    repo_id="SWivid/F5-TTS",
    filename="F5TTS_v1_Base/vocab.txt",
    local_dir=str(MODEL_DIR),
)
sz_f5 = os.path.getsize(f5_model_file) / (1024 * 1024)
print(f"  F5-TTS v1 safetensors : {sz_f5:.1f} MB (downloaded in {time.time()-t0:.1f}s)")
print(f"  path                  : {f5_model_file}")

print("\n== 2. downloading Vocos mel-24khz vocoder ==")
t0 = time.time()
vocos_dir = MODEL_DIR / "vocos-mel-24khz"
vocos_dir.mkdir(parents=True, exist_ok=True)
hf_hub_download(
    repo_id="charactr/vocos-mel-24khz",
    filename="config.yaml",
    local_dir=str(vocos_dir),
)
hf_hub_download(
    repo_id="charactr/vocos-mel-24khz",
    filename="pytorch_model.bin",
    local_dir=str(vocos_dir),
)
print(f"  Vocos vocoder ready in {time.time()-t0:.1f}s at {vocos_dir}")

print("\n== 3. downloading studio reference audio presets ==")
VOICE_URLS = {
    "nature_en.wav": (
        "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
        "Some call me nature, others call me mother nature.",
        "English (Warm Nature / Documentary)",
    ),
    "mandarin_zh.wav": (
        "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
        "对，这就是我，那个一直在默默观察的人。",
        "Mandarin (Clear Conversational)",
    ),
    "storyteller.flac": (
        "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples/multi/country.flac",
        "Once upon a time, there was a little boy who loved nature.",
        "English (Storyteller / Audiobooks)",
    ),
    "narrator.flac": (
        "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples/multi/town.flac",
        "In a quiet little town, surrounded by green hills, lived a family.",
        "English (Calm Narrator)",
    ),
}

manifest = {}
for fname, (url, transcript, description) in VOICE_URLS.items():
    dest = VOICES_DIR / fname
    if not dest.exists() or dest.stat().st_size == 0:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                f.write(resp.read())
        except Exception as e:
            print(f"  warning: failed to download {fname}: {e}")
            continue

    voice_id = dest.stem
    manifest[voice_id] = {
        "file": str(dest),
        "transcript": transcript,
        "description": description,
        "bytes": dest.stat().st_size,
    }
    print(f"  voice preset: {voice_id:15s} | {dest.stat().st_size // 1024} KB | {description}")

manifest_file = VOICES_DIR / "voices.json"
with open(manifest_file, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"  manifest saved: {manifest_file} ({len(manifest)} voices available)")

print("\nDOWNLOAD_OK")
