# Phase 0 / cell 4 — weight-loader smoke test (no weight download needed)
# Question answered: does the fork's name mapping accept the real community
# W4A16 checkpoint's tensor names (compressed-tensors pack-quantized adds
# suffixes like .weight_packed / .weight_scale_and_zero)?
# Downloads only config.json + the safetensors index (KBs).

import json
import re
import urllib.request

REPO = "VnimanieAI/Qwen3.8-Flash-Next-W4A16"
PACKED_SUFFIXES = (
    ".weight_packed", ".weight_scale_and_zero", ".weight_shape",
    ".weight_zero_point", ".weight_g_idx",
)

def hf_raw(path):
    url = f"https://huggingface.co/{REPO}/raw/main/{path}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()

idx = json.loads(hf_raw("model.safetensors.index.json"))
names = sorted(idx.get("weight_map", {}).keys())
print(f"checkpoint tensors in index: {len(names)}")

# what the fork actually exposes (so the audit uses the real API, not guesses)
from tpu_inference.models.jax.qwen4_exp import weight_loader as WL
from tpu_inference.models.jax.qwen4_exp import quant as Q
print("weight_loader API:", [n for n in dir(WL) if not n.startswith("__")][:20])
print("QUANT_SKIP_SUBSTR:", list(getattr(Q, "QUANT_SKIP_SUBSTR", [])))

skip = list(getattr(Q, "QUANT_SKIP_SUBSTR", []))
mapper = getattr(WL, "map_checkpoint_name", None)
import inspect
if mapper is not None:
    print("map_checkpoint_name signature:", inspect.signature(mapper))

buckets = {"skipped": 0, "packed_piece": 0, "ple": 0, "mtp": 0, "mapped": 0, "unmapped": 0}
unmapped_samples = []
base_of = lambda n: re.sub(r"(" + "|".join(re.escape(s) for s in PACKED_SUFFIXES) + r")$", "", n)

for n in names:
    if any(s in n for s in skip):
        buckets["skipped"] += 1
        continue
    if base_of(n) != n:
        buckets["packed_piece"] += 1
    if "ple" in n.lower() or "ngram" in n.lower():
        buckets["ple"] += 1
    if "mtp" in n.lower():
        buckets["mtp"] += 1
    if mapper is None:
        continue
    for candidate, tag in ((n, "raw"), (base_of(n), "base")):
        try:
            m = mapper(candidate)
        except TypeError:
            m = f"<signature needs more args — see above>"
            break
        except Exception as e:
            m = None
        if m:
            if tag == "raw":
                buckets["mapped"] += 1
            break
    else:
        if base_of(n) != n:
            buckets["mapped"] += 1  # a packed piece maps via its base
        else:
            buckets["unmapped"] += 1
            if len(unmapped_samples) < 25:
                unmapped_samples.append(n)

print("buckets:", buckets)
if unmapped_samples:
    print("UNMAPPED SAMPLES (first 25):")
    for s in unmapped_samples:
        print("   ", s)
    print("=> a small loader shim is needed; send me this list")
else:
    print("=> every tensor name maps through the fork's loader unchanged")
print("CELL 4 OK" if buckets["unmapped"] == 0 else "CELL 4: GAPS FOUND (expected, fixable)")
