# CELL 4 — download the Coder GGUF (58.4 GB, 2 shards + mmproj) at full speed.
# hf_transfer = multi-stream (the fast lane); resumable, so re-running the
# cell continues where it stopped. Target: /kaggle/tmp/gguf-coder (1 TB disk).
# ~15-25 min on Kaggle's pipe. The heartbeat keeps the session alive meanwhile.
import os, time

TOKEN = open("/tmp/hf_token").read().strip()
os.environ["HF_TOKEN"] = TOKEN
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
DEST = "/kaggle/tmp/gguf-coder"

from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(REPO, local_dir=DEST, max_workers=8)
dt = time.time() - t0

files = sorted(os.listdir(path))
total = sum(os.path.getsize(f"{path}/{f}") for f in files)
print(f"downloaded {total/1e9:.1f} GB in {dt/60:.1f} min -> {path}")
for f in files:
    print(f"  {f}  {os.path.getsize(f'{path}/{f}')/1e9:.2f} GB")
print("CELL 4 OK — model on disk" if total > 50e9 else "!! download looks incomplete")
