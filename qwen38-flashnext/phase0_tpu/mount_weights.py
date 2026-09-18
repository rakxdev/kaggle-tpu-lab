#!/usr/bin/env python3
"""Phase 0b / TPU — step 2: assemble the checkpoint view from the 13 mounted
datasets + fetch the small meta files from HuggingFace, then verify.

Mounts appear as /kaggle/input/qwen38-flashnext-w4a16-01..13/<shard>. The
loader wants ONE directory: we symlink every shard into /kaggle/tmp/ckpt and
download config/tokenizer/index there (a few MB, plus the 22 MB LFS index).
"""

import json
import os
import urllib.request

CKPT = "/kaggle/tmp/ckpt"
REPO = "VnimanieAI/Qwen3.8-Flash-Next-W4A16"
META = ["config.json", "generation_config.json", "chat_template.jinja",
        "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "preprocessor_config.json"]

os.makedirs(CKPT, exist_ok=True)

# 1. symlink every mounted shard
shards = []
for d in sorted(os.listdir("/kaggle/input")):
    if not d.startswith("qwen38-flashnext-w4a16"):
        continue
    for f in sorted(os.listdir(f"/kaggle/input/{d}")):
        if f.endswith(".safetensors"):
            src = f"/kaggle/input/{d}/{f}"
            dst = f"{CKPT}/{f}"
            if not os.path.exists(dst):
                os.symlink(src, dst)
            shards.append((f, os.path.getsize(src)))
print(f"mounted shards: {len(shards)}  total {sum(s for _, s in shards)/1e9:.1f} GB")
assert len(shards) == 30, f"expected 30 mounted shards, got {len(shards)}"

# 2. meta files from HF (skip if already fetched)
for name in META:
    dst = f"{CKPT}/{name}"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        continue
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    print("fetching", name, flush=True)
    try:
        with urllib.request.urlopen(url, timeout=300) as r, open(dst, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
    except Exception as e:
        print(f"  (skipped {name}: {e})")  # optional files may not exist

# 3. sanity: index maps to existing files
idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))
weight_files = sorted(set(idx.get("weight_map", {}).values()))
missing = [w for w in weight_files if not os.path.exists(f"{CKPT}/{w}")]
print(f"index weight files: {len(weight_files)}, missing from view: {len(missing)}")
assert not missing, f"missing: {missing[:5]}"

print(f"STEP 2 OK — checkpoint view ready at {CKPT}")
