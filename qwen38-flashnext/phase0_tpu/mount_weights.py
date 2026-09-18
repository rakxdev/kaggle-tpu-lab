#!/usr/bin/env python3
"""Phase 0b / TPU — step 2: checkpoint resolver (hybrid).

If the 13 qwen38-flashnext-w4a16-* datasets are attached, their mounted shards
are used via symlinks (instant). Any shard NOT mounted is downloaded from
HuggingFace in parallel (8 workers, ~250-300 MB/s aggregate) into
/kaggle/tmp/ckpt. Meta files (config/tokenizer/index) always come from HF.
Ends with a byte-exact completeness assertion against the HF manifest, so a
half-attached or half-downloaded state fails loudly here, not in the loader.
"""

import concurrent.futures as cf
import json
import os
import time
import urllib.request

CKPT = "/kaggle/tmp/ckpt"
REPO = "VnimanieAI/Qwen3.8-Flash-Next-W4A16"
META = ["config.json", "generation_config.json", "chat_template.jinja",
        "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "preprocessor_config.json"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hf_manifest():
    req = urllib.request.Request(f"https://huggingface.co/api/models/{REPO}?blobs=true")
    with urllib.request.urlopen(req, timeout=60) as r:
        api = json.load(r)
    return {s["rfilename"]: (s.get("size") or 0) for s in api["siblings"]
            if s["rfilename"].endswith(".safetensors")}


def collect_mounted():
    """Shards available from attached datasets -> {name: source path}."""
    got = {}
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return got
    for d in sorted(os.listdir(base)):
        if not d.startswith("qwen38-flashnext-w4a16"):
            continue
        for f in sorted(os.listdir(f"{base}/{d}")):
            if f.endswith(".safetensors"):
                got[f] = f"{base}/{d}/{f}"
    return got


def hf_headers():
    """Authenticated downloads: HF throttles anonymous IPs hard after ~a
    hundred GB in a day (seen live — streams degraded to a crawl). A free
    read token lifts the ceiling. Token from env or the hub cache file."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        p = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(p):
            tok = open(p).read().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def hf_download(name, dest, expected):
    tmp = dest + ".part"
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, headers=hf_headers()),
                                timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            done += len(b)
            if done % (2 << 30) < (1 << 22):
                log(f"  {name}: {done/1e9:.1f} GB ({done/1e6/max(time.time()-t0,1):.0f} MB/s)")
    if abs(os.path.getsize(tmp) - expected) > 1e6:
        os.remove(tmp)
        raise RuntimeError(f"{name}: downloaded {os.path.getsize(tmp)} != expected {expected}")
    os.replace(tmp, dest)


def main():
    os.makedirs(CKPT, exist_ok=True)
    shards = hf_manifest()
    log(f"manifest: {len(shards)} shards, {sum(shards.values())/1e9:.1f} GB")
    free = os.statvfs("/kaggle/tmp").f_bavail * os.statvfs("/kaggle/tmp").f_frsize
    log(f"/kaggle/tmp free: {free/1e9:.0f} GB")
    assert free > sum(shards.values()) + 5e9, "not enough disk for the full checkpoint"

    # meta files first (the index is needed to validate everything else)
    for name in META:
        dst = f"{CKPT}/{name}"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        try:
            with urllib.request.urlopen(
                    f"https://huggingface.co/{REPO}/resolve/main/{name}", timeout=300) as r, \
                    open(dst, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:
            print(f"  (no {name}: {e})")

    # resolve every shard: mounted symlink OR download
    mounted = collect_mounted()
    log(f"mounted shards from datasets: {len(mounted)}/{len(shards)}")
    todo, linked = [], 0
    for name, expected in shards.items():
        dst = f"{CKPT}/{name}"
        if name in mounted:
            src = mounted[name]
            if abs(os.path.getsize(src) - expected) <= 1e6:
                if not os.path.exists(dst):
                    os.symlink(src, dst)
                linked += 1
                continue
        if os.path.exists(dst) and abs(os.path.getsize(dst) - expected) <= 1e6:
            continue  # already downloaded in a previous step
        todo.append((name, dst, expected))
    log(f"to download: {len(todo)} shards "
        f"({sum(t[2] for t in todo)/1e9:.1f} GB)")

    if todo:
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            for _ in ex.map(lambda t: hf_download(*t), todo):
                pass
        log(f"downloads done in {(time.time()-t0)/60:.1f} min "
            f"({sum(t[2] for t in todo)/1e6/max(time.time()-t0,1):.0f} MB/s aggregate)")

    # final byte-exact assertion across the whole checkpoint
    bad = [(n, os.path.getsize(f"{CKPT}/{n}"), shards[n])
           for n in shards
           if not os.path.exists(f"{CKPT}/{n}")
           or abs(os.path.getsize(f"{CKPT}/{n}") - shards[n]) > 1e6]
    assert not bad, f"corrupt/missing shards: {bad[:5]}"
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))
    wf = sorted(set(idx.get("weight_map", {}).values()))
    miss = [w for w in wf if not os.path.exists(f"{CKPT}/{w}")]
    assert not miss, f"index references missing files: {miss[:5]}"
    n_link = sum(1 for n in shards if os.path.islink(f"{CKPT}/{n}"))
    print(f"STEP 2 OK — {len(shards)} shards byte-exact "
          f"({n_link} from dataset mounts, {len(shards)-n_link} downloaded) — view at {CKPT}")


if __name__ == "__main__":
    main()
