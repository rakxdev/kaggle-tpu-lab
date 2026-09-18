#!/usr/bin/env python3
"""Pack the Qwen3.8-Flash-Next W4A16 checkpoint into Kaggle datasets.

Downloads shards from HuggingFace in pairs (each pair <= ~18 GB, safely under
the 20 GB /kaggle/working cap), creates one public dataset per pair via the
kaggle CLI, deletes the local copy, and continues. Idempotent per dataset:
existing ones get a new version instead of failing. Runs as a detached job —
progress goes to /kaggle/working/pack.log.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = "VnimanieAI/Qwen3.8-Flash-Next-W4A16"
OWNER = "rakeshbehera42"
PREFIX = "qwen38-flashnext-w4a16"
WORK = "/kaggle/working/pack"
BIN_LIMIT = 18.0e9  # per-dataset byte budget (pair the 30 shards -> ~15 datasets)
LICENSES = [{"name": "other"}]


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
    return r


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shard_sizes():
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{REPO}?blobs=true")
    with urllib.request.urlopen(req, timeout=60) as r:
        api = json.load(r)
    files = {s["rfilename"]: (s.get("size") or 0) for s in api["siblings"]}
    return sorted(
        ((n, sz) for n, sz in files.items()
         if n.endswith(".safetensors") and sz > 1e6),
        key=lambda t: t[0])


def download(name, dest):
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    tmp = dest + ".part"
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while True:
            block = r.read(8 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if done % (2 << 30) < (8 << 20):
                log(f"  {os.path.basename(dest)}: {done/1e9:.1f} GB "
                    f"({done/1e6/max(time.time()-t0,1):.0f} MB/s)")
    os.replace(tmp, dest)


def create_dataset(folder, ds_id):
    meta = {"id": ds_id, "title": ds_id.split("/")[1], "licenses": LICENSES}
    json.dump(meta, open(os.path.join(folder, "dataset-metadata.json"), "w"))
    # drop any leftover from an earlier attempt (private or partial) — a
    # dataset version would keep its old visibility
    sh(f"kaggle datasets delete -y {ds_id}", check=False)
    out = ""
    for attempt in (1, 2):
        r = sh(f"kaggle datasets create -p {folder} -u", check=False)  # -u = public
        out = r.stdout + r.stderr
        if "being created" in out or "successfully" in out:
            return "created-public"
        log(f"  create attempt {attempt} failed: {out[-220:]!r}")
        time.sleep(30)
    return f"create-failed: {out[-220:]}"


def main():
    os.makedirs(WORK, exist_ok=True)
    shards = shard_sizes()
    log(f"{len(shards)} shards, total {sum(s for _, s in shards)/1e9:.1f} GB")

    # greedy bin consecutive shards under the byte budget
    bins, cur, cur_sz = [], [], 0
    for name, sz in shards:
        if cur and cur_sz + sz > BIN_LIMIT:
            bins.append(cur)
            cur, cur_sz = [], 0
        cur.append((name, sz))
        cur_sz += sz
    if cur:
        bins.append(cur)
    log(f"{len(bins)} datasets planned: "
        + ", ".join(f"{sum(s for _, s in b)/1e9:.1f}" for b in bins) + " GB")

    results = []
    for n, files in enumerate(bins, 1):
        ds_id = f"{OWNER}/{PREFIX}-{n:02d}"
        folder = os.path.join(WORK, f"chunk{n:02d}")
        t0 = time.time()
        log(f"=== dataset {n}/{len(bins)}: {ds_id} "
            f"({sum(s for _, s in files)/1e9:.1f} GB) ===")
        os.makedirs(folder, exist_ok=True)
        try:
            from concurrent.futures import ThreadPoolExecutor
            targets = []
            for name, _ in files:
                dest = os.path.join(folder, os.path.basename(name))
                if not (os.path.exists(dest) and abs(os.path.getsize(dest) - dict(shards)[name]) < 1e6):
                    targets.append((name, dest))
            if targets:
                log(f"  downloading {len(targets)} shards in parallel")
                with ThreadPoolExecutor(max_workers=min(3, len(targets))) as ex:
                    for r in ex.map(lambda t: download(*t), targets):
                        pass
            res = create_dataset(folder, ds_id)
        except Exception as e:
            res = f"error: {e!r}"[:300]
        results.append((ds_id, res))
        log(f"  -> {res} ({(time.time()-t0)/60:.1f} min)")
        shutil.rmtree(folder, ignore_errors=True)  # free the 20 GB cap

    log("=== SUMMARY ===")
    for ds_id, res in results:
        log(f"  {ds_id}: {res}")
    ok = sum(1 for _, r in results if r in ("created", "versioned"))
    log(f"{ok}/{len(results)} datasets published")


if __name__ == "__main__":
    main()
