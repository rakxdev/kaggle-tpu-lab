#!/usr/bin/env python3
"""
CELL 3 — Download IndexTTS-2.5 model weights and example voice/emotion audio prompts.

Downloads:
1. IndexTeam/IndexTTS-2.5 checkpoints from Hugging Face into /kaggle/tmp/indextts/checkpoints
2. Official example reference audio and emotion audio prompts
"""

import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("!! huggingface_hub missing, run cell 0 / 2 first", file=sys.stderr)
    sys.exit(1)

BASE_DIR = Path("/kaggle/tmp/indextts")
CHECKPOINTS = BASE_DIR / "checkpoints"
EXAMPLES_DIR = BASE_DIR / "examples"
LOG_DIR = BASE_DIR / "logs"

for p in (CHECKPOINTS, EXAMPLES_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

print("== 1. downloading IndexTTS-2.5 model checkpoints ==")
t0 = time.time()
snapshot_download(
    repo_id="IndexTeam/IndexTTS-2.5",
    local_dir=str(CHECKPOINTS),
    max_workers=8,
)
dt = time.time() - t0
print(f"  checkpoints ready in {dt:.1f}s at {CHECKPOINTS}")

print("\n== 2. downloading official voice & emotion example clips ==")
# Import repo helper to download examples if available
repo_path = BASE_DIR / "repo"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

try:
    from indextts.utils.examples_downloader import ensure_examples_available
    ensure_examples_available()
    print("  official example audio clips downloaded via indextts helper")
except Exception as e:
    print(f"  examples downloader note: {e}")

# Check files
print("\n== 3. verifying downloaded files ==")
required_files = ["config.yaml", "gpt.pth", "s2mel.pth", "codec.pth"]
for rf in required_files:
    fpath = CHECKPOINTS / rf
    if fpath.exists():
        sz_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  ✓ {rf:15s} : {sz_mb:.1f} MB")
    else:
        print(f"  !! MISSING: {rf}")

print("\nDOWNLOAD_OK")
