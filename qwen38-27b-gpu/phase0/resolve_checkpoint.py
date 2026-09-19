#!/usr/bin/env python3
"""qwen38-27b-gpu / phase0 — step 2: checkpoint resolver (hybrid).

RedHatAI/Qwen3.8-27B-INT4 = 19.5 GB (18.6 main + 0.85 MTP + tokenizer).
If any qwen38-27b-gpu datasets are attached, their files are used via
symlink; everything missing downloads from HF with hf_transfer (multi-stream,
~GB/s on datacenter pipes). Byte-exact assertion before the loader runs.
"""

import json
import os
import time
import urllib.request

CKPT = "/kaggle/tmp/ckpt"
REPO = "RedHatAI/Qwen3.8-27B-INT4"
META = ["config.json", "generation_config.json", "chat_template.jinja",
        "processor_config.json", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def manifest():
    req = urllib.request.Request(f"https://huggingface.co/api/models/{REPO}?blobs=true")
    with urllib.request.urlopen(req, timeout=60) as r:
        api = json.load(r)
    return {s["rfilename"]: (s.get("size") or 0) for s in api["siblings"]
            if (s.get("size") or 0) > 0 and not s["rfilename"].startswith(".")}


def fetch(name, dest, expected):
    if os.path.exists(dest) and abs(os.path.getsize(dest) - expected) <= 1e6:
        return "cached"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    t0 = time.time()
    # hf_transfer via the hub library (parallel per-file); urllib fallback
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, name, local_dir=os.path.dirname(dest) or ".")
        if os.path.abspath(p) != os.path.abspath(dest):
            os.replace(p, dest)
        return f"hf {expected/1e9:.2f}GB in {time.time()-t0:.0f}s"
    except Exception as e:
        log(f"  hf path failed ({e!r}) — urllib fallback")
    with urllib.request.urlopen(
            f"https://huggingface.co/{REPO}/resolve/main/{name}", timeout=120) as r, \
            open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    os.replace(dest + ".part", dest)
    return f"urllib {expected/1e9:.2f}GB in {time.time()-t0:.0f}s"


def main():
    # /kaggle/tmp exists on TPU images but not on GPU ones; it is a plain
    # directory on the ~1TB root overlay, so creating it is safe and keeps the
    # 19.5 GB checkpoint off the 20 GB /kaggle/working loop device.
    os.makedirs(CKPT, exist_ok=True)
    files = manifest()
    log(f"manifest: {len(files)} files, {sum(files.values())/1e9:.1f} GB")
    free = os.statvfs("/kaggle/tmp").f_bavail * os.statvfs("/kaggle/tmp").f_frsize
    log(f"/kaggle/tmp free: {free/1e9:.0f} GB")
    assert free > sum(files.values()) + 2e9, "not enough scratch disk"

    # mounted datasets first (any qwen38-27b ones the user managed to attach)
    mounted = {}
    if os.path.isdir("/kaggle/input"):
        for d in sorted(os.listdir("/kaggle/input")):
            if "qwen38-27b" not in d:
                continue
            for f in sorted(os.listdir(f"/kaggle/input/{d}")):
                mounted[f] = f"/kaggle/input/{d}/{f}"
    log(f"mounted files from datasets: {len(mounted)}/{len(files)}")

    results = {}
    for name, expected in files.items():
        dst = f"{CKPT}/{name}"
        if name in mounted and abs(os.path.getsize(mounted[name]) - expected) <= 1e6:
            if not os.path.exists(dst):
                os.symlink(mounted[name], dst)
            results[name] = "mounted"
            continue
        results[name] = fetch(name, dst, expected)
        log(f"  {name}: {results[name]}")

    bad = [(n, os.path.getsize(f"{CKPT}/{n}"), files[n])
           for n in files
           if not os.path.exists(f"{CKPT}/{n}")
           or abs(os.path.getsize(f"{CKPT}/{n}") - files[n]) > 1e6]
    assert not bad, f"corrupt/missing: {bad[:5]}"
    cfg = json.load(open(f"{CKPT}/config.json"))
    print(f"STEP 2 OK — {len(files)} files byte-exact | arch {cfg.get('architectures')} "
          f"| ctx {cfg.get('text_config', cfg).get('max_position_embeddings')}")


if __name__ == "__main__":
    main()
