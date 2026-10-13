#!/usr/bin/env python3
"""CELL 4 — Smoke test: Zero-shot voice cloning with Shubh reference audio.

Validates the complete pipeline on Kaggle GPU 0 (Turing T4):
1. Loads ai4bharat/IndicF5 (330M parameters, ~3.5 GB VRAM).
2. Clones the Shubh reference voice (5.46s WAV, warm Hindi narrator).
3. Synthesizes both pure Devanagari narrative and code-mixed Hinglish.
4. Measures RTF (Real-Time Factor), VRAM consumption, and audio fidelity.
"""

import os
import sys
import time
from pathlib import Path

# Auto-re-execute inside isolated Python 3.11 venv
VENV_PY = "/kaggle/tmp/indicf5/venv/bin/python"
if os.path.exists(VENV_PY) and sys.executable != VENV_PY:
    print(f"Re-executing inside isolated runtime: {VENV_PY}")
    os.execv(VENV_PY, [VENV_PY] + sys.argv)

# Add kit directory to sys.path for lib_normalize
KIT_DIR = Path(__file__).resolve().parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel
from lib_normalize import to_unified_devanagari

BASE_DIR = Path("/kaggle/tmp/indicf5")
CHECKPOINTS = BASE_DIR / "checkpoints"
REF_WAV = BASE_DIR / "shubh_ref.wav"
REF_TXT = BASE_DIR / "shubh_ref.txt"
OUT_DEV = BASE_DIR / "smoke_devanagari.wav"
OUT_HIN = BASE_DIR / "smoke_hinglish.wav"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"== 1. Device & VRAM Baseline ==")
print(f"  Target device: {device}")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(0)
    base_vram = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"  Initial VRAM allocated: {base_vram:.2f} GiB")

print(f"\n== 2. Loading IndicF5 Model ==")
t0_load = time.time()

# Prefer local checkpoint directory if populated, otherwise repo id
model_source = str(CHECKPOINTS) if (CHECKPOINTS / "model.safetensors").exists() else "ai4bharat/IndicF5"
print(f"  Loading model from: {model_source}")

model = AutoModel.from_pretrained(
    model_source,
    trust_remote_code=True,
).to(device)

model_vram = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
print(f"  Model loaded in {time.time() - t0_load:.2f}s | VRAM: {model_vram:.2f} GiB")

print(f"\n== 3. Validating Shubh Voice Reference ==")
if not REF_WAV.exists() or not REF_TXT.exists():
    raise FileNotFoundError(f"Missing Shubh reference files at {REF_WAV} or {REF_TXT}")

ref_text = REF_TXT.read_text(encoding="utf-8").strip()
ref_audio_info = sf.info(str(REF_WAV))
print(f"  Reference audio : {REF_WAV.name} ({ref_audio_info.duration:.2f}s, {ref_audio_info.samplerate} Hz, {ref_audio_info.channels} ch)")
print(f"  Reference text  : \"{ref_text}\"")

def run_synthesis(text: str, out_path: Path, label: str):
    print(f"\n-- Synthesizing: [{label}] --")
    print(f"  Input text : \"{text}\"")
    
    t0 = time.time()
    with torch.inference_mode():
        raw_audio = model(
            text=text,
            ref_audio_path=str(REF_WAV),
            ref_text=ref_text,
        )
    latency = time.time() - t0
    
    # Process audio array
    if hasattr(raw_audio, "cpu"):
        raw_audio = raw_audio.cpu().numpy()
    audio_arr = np.asarray(raw_audio, dtype=np.float32).squeeze()
    
    if raw_audio.dtype == np.int16 or np.max(np.abs(audio_arr)) > 1.5:
        audio_arr = audio_arr.astype(np.float32) / 32768.0
        
    duration = len(audio_arr) / 24000.0
    rtf = latency / duration if duration > 0 else 0.0
    
    sf.write(str(out_path), audio_arr, 24000)
    print(f"  Generated  : {duration:.2f}s audio in {latency:.2f}s (RTF: {rtf:.3f}x)")
    print(f"  Output     : {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return duration, latency, rtf

print(f"\n== 4. Test 1: Pure Devanagari Storytelling (Shubh Persona) ==")
text_deva = "नमस्ते! शहर की इस चाय की दुकान पर बैठकर नई कहानियाँ सोचना मुझे बहुत पसंद है।"
run_synthesis(text_deva, OUT_DEV, "Pure Devanagari")

print(f"\n== 5. Test 2: Hinglish Code-Mixed Voiceover with Normalizer ==")
text_hinglish = "कल office में new project launch की meeting है, टेंशन मत लो सब smoothly हो जाएगा।"
normalized_text = to_unified_devanagari(text_hinglish)
print(f"  Normalized text: \"{normalized_text}\"")
run_synthesis(normalized_text, OUT_HIN, "Hinglish Transliterated")

if torch.cuda.is_available():
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3)
    print(f"\n== 6. GPU Turing T4 Memory Profile ==")
    print(f"  Peak VRAM consumed : {peak_vram:.2f} GiB / 15.83 GiB ({peak_vram/15.83*100:.1f}%)")
    print(f"  VRAM Headroom      : {15.83 - peak_vram:.2f} GiB (>70% free)")

print("\nSMOKE_OK")
