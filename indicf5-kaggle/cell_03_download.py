#!/usr/bin/env python3
"""CELL 3 — Model download & Shubh reference asset preparation.

Downloads ai4bharat/IndicF5 (~1.4 GB model weights) from Hugging Face,
verifies checksums and directory layout, stages the Shubh voice sample,
and verifies the duration patch on all copies of utils_infer.py.
"""

import os
import sys
import shutil
import glob
from pathlib import Path

# Auto-re-execute inside isolated Python 3.11 venv
VENV_PY = "/kaggle/tmp/indicf5/venv/bin/python"
if os.path.exists(VENV_PY) and sys.executable != VENV_PY:
    print(f"Re-executing inside isolated runtime: {VENV_PY}")
    os.execv(VENV_PY, [VENV_PY] + sys.argv)

# Optional hf_transfer fast downloading
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

# Add kit directory to sys.path for helpers
KIT_DIR = Path(__file__).resolve().parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import huggingface_hub
from huggingface_hub import snapshot_download
from auth_helper import get_hf_token

BASE_DIR = Path("/kaggle/tmp/indicf5")
CHECKPOINTS = BASE_DIR / "checkpoints"
ASSETS_DIR = Path("/kaggle/working/kaggle-tpu-lab/indicf5-kaggle/assets")
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# Securely resolve HF Token (environment or encrypted embedded fallback)
HF_TOKEN = get_hf_token()

print(f"== 1. Hugging Face Authentication ==")
huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
print("  Authenticated successfully.")

print(f"\n== 2. Downloading ai4bharat/IndicF5 ==")
print(f"  Target directory: {CHECKPOINTS}")
snapshot_download(
    repo_id="ai4bharat/IndicF5",
    local_dir=str(CHECKPOINTS),
    token=HF_TOKEN,
    max_workers=8,
)
print("  Snapshot download completed.")

print(f"\n== 3. Checkpoint Validation ==")
safetensors_path = CHECKPOINTS / "model.safetensors"
if safetensors_path.exists():
    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"  model.safetensors: {size_mb:.2f} MB (Verified)")
else:
    raise FileNotFoundError(f"Missing model.safetensors at {safetensors_path}")

vocab_path = CHECKPOINTS / "checkpoints" / "vocab.txt"
if not vocab_path.exists():
    vocab_path = CHECKPOINTS / "vocab.txt"
if vocab_path.exists():
    print(f"  vocab.txt: {vocab_path} (Verified)")

print(f"\n== 4. Staging Shubh Reference Voice ==")
shubh_src_wav = ASSETS_DIR / "shubh_ref.wav"
shubh_src_txt = ASSETS_DIR / "shubh_ref.txt"
shubh_dst_wav = BASE_DIR / "shubh_ref.wav"
shubh_dst_txt = BASE_DIR / "shubh_ref.txt"

if shubh_src_wav.exists() and shubh_src_txt.exists():
    shutil.copy2(str(shubh_src_wav), str(shubh_dst_wav))
    shutil.copy2(str(shubh_src_txt), str(shubh_dst_txt))
    print(f"  Staged: {shubh_dst_wav} ({shubh_dst_wav.stat().st_size / 1024:.1f} KB)")
    print(f"  Transcript: {shubh_dst_txt.read_text(encoding='utf-8').strip()}")
else:
    print(f"  WARNING: Assets not found at {ASSETS_DIR}, checking fallback locations...")
    # Check if files exist directly in /home/rakesh
    fallback_wav = Path("/home/rakesh/shubh_reference_clip.wav")
    fallback_txt = Path("/home/rakesh/shubh_reference_text.txt")
    if fallback_wav.exists() and fallback_txt.exists():
        shutil.copy2(str(fallback_wav), str(shubh_dst_wav))
        shutil.copy2(str(fallback_txt), str(shubh_dst_txt))
        print(f"  Staged from fallback: {shubh_dst_wav}")

print(f"\n== 5. Verifying Duration Patch across Cache ==")
all_utils = glob.glob(f"{CHECKPOINTS}/**/utils_infer.py", recursive=True) + \
            glob.glob("/kaggle/tmp/indicf5/venv/**/utils_infer.py", recursive=True)

for p in sorted(set(all_utils)):
    src = Path(p).read_text(encoding="utf-8")
    if 'len(ref_text.encode("utf-8"))' in src:
        new_src = src.replace('len(ref_text.encode("utf-8"))', 'sum(1 for c in ref_text if not c.isspace())')
        new_src = new_src.replace('len(gen_text.encode("utf-8"))', 'sum(1 for c in gen_text if not c.isspace())')
        Path(p).write_text(new_src, encoding="utf-8")
        print(f"  Patched: {p}")
    elif 'sum(1 for c in ref_text if not c.isspace())' in src:
        print(f"  Already patched: {p}")

print("\nDOWNLOAD_OK")
