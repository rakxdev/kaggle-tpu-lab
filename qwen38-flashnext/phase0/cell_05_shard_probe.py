# Phase 0 / cell 5 — read every shard's safetensors header (no weight download)
# Two things answered with a few MB of HTTP range requests per shard:
#   1. the real HBM-vs-host split: how many GB are model (goes on the 8 chips)
#      vs n-gram/PLE table (lives in host RAM)
#   2. the exact shard -> dataset grouping for the packaging step that follows

import json
import struct
import urllib.request

REPO = "VnimanieAI/Qwen3.8-Flash-Next-W4A16"


def hf_url(path):
    return f"https://huggingface.co/{REPO}/resolve/main/{path}"


def hf_json(path):
    with urllib.request.urlopen(hf_url(path), timeout=30) as r:
        return json.load(r)


api = json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/models/{REPO}", timeout=30))
shards = sorted(s["rfilename"] for s in api["siblings"]
                if s["rfilename"].endswith(".safetensors"))
print(f"shards: {len(shards)}")


def header_of(shard):
    """safetensors = u64 header_len + JSON header. Two tiny range requests."""
    req = urllib.request.Request(hf_url(shard), headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=30) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(hf_url(shard), headers={"Range": f"bytes=8-{7 + n}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


cat_bytes = {"model_packed_u8": 0, "model_scales_fp": 0, "model_other_fp": 0,
             "ple_ngram": 0, "mtp": 0}
shard_totals = {}

for shard in shards:
    hdr = header_of(shard)
    total = 0
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        dt = meta.get("dtype", "?")
        shape = meta.get("shape", [])
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
                   "I8": 1, "U8": 1, "F8_E4M3": 1}.get(dt, 1)
        total += nbytes
        ln = name.lower()
        if "ple" in ln or "ngram" in ln:
            cat_bytes["ple_ngram"] += nbytes
        elif "mtp" in ln:
            cat_bytes["mtp"] += nbytes
        elif dt == "U8" or "packed" in ln:
            cat_bytes["model_packed_u8"] += nbytes
        elif dt in ("BF16", "F16", "F32"):
            cat_bytes["model_scales_fp" if "scale" in ln else "model_other_fp"] += nbytes
        else:
            cat_bytes["model_other_fp"] += nbytes
    shard_totals[shard] = total

print(f"\n{'shard':<28} {'GB':>7}")
for s, b in sorted(shard_totals.items()):
    print(f"{s:<28} {b/1e9:7.2f}")
print(f"\nTOTAL: {sum(shard_totals.values())/1e9:.1f} GB")
print("\nper category (this decides the memory plan):")
for k, b in cat_bytes.items():
    print(f"  {k:<18} {b/1e9:8.2f} GB")

main = cat_bytes["model_packed_u8"] + cat_bytes["model_scales_fp"] + cat_bytes["model_other_fp"]
print(f"\n=> HBM-resident (model): {main/1e9:.1f} GB  -> {(main/1e9)/8:.2f} GB per chip")
print(f"=> host RAM (PLE/n-gram): {cat_bytes['ple_ngram']/1e9:.1f} GB"
      f"  (+MTP {cat_bytes['mtp']/1e9:.1f} GB)")
print("CELL 5 OK")
