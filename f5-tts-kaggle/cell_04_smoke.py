#!/usr/bin/env python3
"""
CELL 4 — Smoke test for F5-TTS v1 zero-shot voice cloning on T4 GPU.

Loads the model onto CUDA:0, clones a voice from a 5-second reference clip,
generates a speech WAV, measures synthesis latency and Real-Time Factor (RTF),
and checks peak GPU memory consumption.
"""

import os
import sys
import time
import json
from pathlib import Path

BASE_DIR = Path("/kaggle/tmp/f5tts")
MODEL_DIR = BASE_DIR / "models"
VOICES_DIR = BASE_DIR / "voices"
OUT_WAV = BASE_DIR / "smoke_out.wav"

print("== 1. environment & GPU preflight ==")
try:
    import torch
    import torchaudio
    from f5_tts.api import F5TTS
except ImportError as e:
    print(f"!! dependency import failed: {e}", file=sys.stderr)
    print("   Please ensure Cell 0 and Cell 2 completed successfully.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("!! CUDA not available. Running F5-TTS requires GPU (T4).", file=sys.stderr)
    sys.exit(1)

device = "cuda:0"
gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"  Target Device : {device} ({gpu_name}, {total_vram:.2f} GiB total VRAM)")

print("\n== 2. loading F5-TTS v1 model & vocoder ==")
ckpt_file = MODEL_DIR / "F5TTS_v1_Base" / "model_1250000.safetensors"
vocab_file = MODEL_DIR / "F5TTS_v1_Base" / "vocab.txt"

t_load_start = time.time()
f5tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=str(ckpt_file) if ckpt_file.exists() else "",
    vocab_file=str(vocab_file) if vocab_file.exists() else "",
    device=device,
)
t_load_end = time.time()
print(f"  Model loaded into VRAM in {t_load_end - t_load_start:.2f}s")
vram_after_load = torch.cuda.memory_allocated(0) / (1024**3)
print(f"  Base Model VRAM usage: {vram_after_load:.2f} GiB")

print("\n== 3. selecting zero-shot reference voice clip ==")
manifest_file = VOICES_DIR / "voices.json"
if manifest_file.exists():
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    ref_voice = manifest.get("nature_en") or next(iter(manifest.values()))
    ref_file = ref_voice["file"]
    ref_text = ref_voice["transcript"]
    desc = ref_voice.get("description", "Preset voice")
else:
    from importlib.resources import files
    ref_file = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    ref_text = "Some call me nature, others call me mother nature."
    desc = "Default package reference"

print(f"  Reference file : {ref_file}")
print(f"  Reference text : {ref_text!r} ({desc})")

print("\n== 4. executing zero-shot speech synthesis ==")
gen_text = "Welcome to open-source neural audio synthesis. This is F5-TTS running live on NVIDIA T4."
print(f"  Target text    : {gen_text!r}")

torch.cuda.reset_peak_memory_stats(0)
t0 = time.time()
wav, sr, spec = f5tts.infer(
    ref_file=ref_file,
    ref_text=ref_text,
    gen_text=gen_text,
    file_wave=str(OUT_WAV),
    nfe_step=32,
    speed=1.0,
    seed=42,
)
dt = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3)

audio_duration = len(wav) / sr if hasattr(wav, "__len__") else 0.0
rtf = dt / max(audio_duration, 1e-6)

print(f"\n== 5. performance report ==")
print(f"  Output audio file    : {OUT_WAV} ({OUT_WAV.stat().st_size:,} bytes)")
print(f"  Sample rate          : {sr} Hz")
print(f"  Generated duration   : {audio_duration:.2f} seconds")
print(f"  Synthesis wall clock : {dt:.2f} seconds")
print(f"  Real-Time Factor     : {rtf:.3f}x ({1.0/max(rtf, 1e-6):.1f}x faster than real-time)")
print(f"  Peak VRAM consumption: {peak_vram:.2f} GiB (leaves {total_vram - peak_vram:.2f} GiB headroom on GPU 0)")

assert OUT_WAV.exists() and OUT_WAV.stat().st_size > 10000, "Output WAV file was not generated properly"

print("\nSMOKE_OK")
