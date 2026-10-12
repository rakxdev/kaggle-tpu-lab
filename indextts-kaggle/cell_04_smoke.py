#!/usr/bin/env python3
"""
CELL 4 — Smoke test for IndexTTS-2.5 on T4 GPU:
1. Loads IndexTTS2 engine in FP16 mode on CUDA:0
2. Executes zero-shot voice cloning
3. Executes emotional voice cloning with 8-axis emotion vector
4. Measures VRAM allocation and synthesis latency
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path("/kaggle/tmp/indextts")
CHECKPOINTS = BASE_DIR / "checkpoints"
REPO_DIR = BASE_DIR / "repo"
OUT_NEUTRAL = BASE_DIR / "smoke_neutral.wav"
OUT_EMOTION = BASE_DIR / "smoke_emotion.wav"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print("== 1. environment & GPU preflight ==")
try:
    import torch
    import torchaudio
    from indextts.infer_v2_5 import IndexTTS2
except ImportError as e:
    print(f"!! dependency import failed: {e}", file=sys.stderr)
    print("   Please ensure Cell 0 and Cell 2 completed successfully.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("!! CUDA not available. Running IndexTTS-2.5 requires GPU (T4).", file=sys.stderr)
    sys.exit(1)

device = "cuda:0"
gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"  Target Device : {device} ({gpu_name}, {total_vram:.2f} GiB total VRAM)")

print("\n== 2. loading IndexTTS-2.5 engine (FP16) ==")
cfg_path = CHECKPOINTS / "config.yaml"
t0 = time.time()

tts = IndexTTS2(
    cfg_path=str(cfg_path),
    model_dir=str(CHECKPOINTS),
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False,
)
print(f"  IndexTTS-2.5 loaded in {time.time() - t0:.2f}s")
vram_loaded = torch.cuda.memory_allocated(0) / (1024**3)
print(f"  Base Model VRAM usage: {vram_loaded:.2f} GiB")

print("\n== 3. discovering reference voice audio ==")
# Search for available voice sample
candidates = [
    REPO_DIR / "examples" / "voice_01.wav",
    BASE_DIR / "examples" / "voice_01.wav",
    Path("/kaggle/tmp/f5tts/voices/nature_en.wav"),
]
ref_audio = None
for c in candidates:
    if c.exists() and c.stat().st_size > 1000:
        ref_audio = str(c)
        break

if not ref_audio:
    # Generate quick synthetic test reference if needed
    ref_audio = str(BASE_DIR / "ref_temp.wav")
    import numpy as np, soundfile as sf
    sf.write(ref_audio, np.sin(2 * np.pi * 440 * np.linspace(0, 4, 4 * 24000)), 24000)

print(f"  Reference audio : {ref_audio}")

print("\n== 4. test A: standard zero-shot voice cloning ==")
text_a = "Hello, this is IndexTTS-2.5 running with high-fidelity voice cloning on NVIDIA T4."
torch.cuda.reset_peak_memory_stats(0)
t0 = time.time()
tts.infer(
    spk_audio_prompt=ref_audio,
    text=text_a,
    lang="EN",
    output_path=str(OUT_NEUTRAL),
    verbose=False,
)
dt_a = time.time() - t0
peak_a = torch.cuda.max_memory_allocated(0) / (1024**3)
print(f"  Generated in {dt_a:.2f}s | Output: {OUT_NEUTRAL} ({OUT_NEUTRAL.stat().st_size:,} bytes)")

print("\n== 5. test B: emotional synthesis with emotion vector ==")
# Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
emo_vector = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0] # 90% Happy + 50% Surprised
text_b = "I cannot believe this! We actually won the entire championship!"
t0 = time.time()
tts.infer(
    spk_audio_prompt=ref_audio,
    text=text_b,
    lang="EN",
    emo_vector=emo_vector,
    emo_alpha=0.9,
    output_path=str(OUT_EMOTION),
    verbose=False,
)
dt_b = time.time() - t0
peak_b = torch.cuda.max_memory_allocated(0) / (1024**3)
print(f"  Emotional audio generated in {dt_b:.2f}s | Output: {OUT_EMOTION} ({OUT_EMOTION.stat().st_size:,} bytes)")
print(f"  Peak VRAM: {peak_b:.2f} GiB (leaves {total_vram - peak_b:.2f} GiB headroom on GPU 0)")

assert OUT_NEUTRAL.exists() and OUT_NEUTRAL.stat().st_size > 10000, "Neutral WAV generation failed"
assert OUT_EMOTION.exists() and OUT_EMOTION.stat().st_size > 10000, "Emotional WAV generation failed"

print("\nSMOKE_OK")
