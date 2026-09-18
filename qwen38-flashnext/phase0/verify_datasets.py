#!/usr/bin/env python3
"""Verify the packed datasets: every expected shard present with matching size,
and every dataset public. Compares kaggle's file listing (name + size) against
the HuggingFace source sizes for all planned dataset bins."""

import json
import os
import re
import urllib.request

import pack_datasets as P  # same binning logic the packer used


def hf_sizes():
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{P.REPO}?blobs=true")
    with urllib.request.urlopen(req, timeout=60) as r:
        api = json.load(r)
    return {s["rfilename"]: (s.get("size") or 0) for s in api["siblings"]
            if s["rfilename"].endswith(".safetensors")}  # same filter as the packer


def kaggle_files(ds_id):
    r = P.sh(f"kaggle datasets files {ds_id}", check=False)
    out = {}
    for line in r.stdout.splitlines():
        s = line.strip()
        if not s or set(s) <= set("- ") or s.lower().startswith(("name", "ref", "size")):
            continue
        parts = s.split()
        name, size_gb = parts[0], 0.0
        for tok in parts[1:]:
            m = re.match(r"^([0-9.]+)(GB|MB|KB|B)$", tok, re.I)
            if m:
                mult = {"GB": 1e9, "MB": 1e6, "KB": 1e3, "B": 1}[m.group(2).upper()]
                size_gb = float(m.group(1)) * mult / 1e9
                break
            try:
                size_gb = int(tok) / 1e9  # raw bytes, e.g. 6400032032
                break
            except ValueError:
                continue
        out[name] = size_gb
    return out


def kaggle_private_flags():
    req = urllib.request.Request(
        "https://www.kaggle.com/api/v1/datasets/list"
        f"?user={P.OWNER}&pageSize=50",
        headers={"Authorization": "Bearer " + open(os.path.expanduser(
            "~/.kaggle/access_token")).read().strip()})
    with urllib.request.urlopen(req, timeout=60) as r:
        ds = json.load(r)
    return {d.get("refNullable") or d.get("ref"):
            d.get("isPrivateNullable", d.get("isPrivate")) for d in ds}


def main():
    shards = hf_sizes()
    flags = kaggle_private_flags()
    # same greedy binning as the packer
    names = sorted((n, s) for n, s in shards.items() if s > 1e6)
    bins, cur, cur_sz = [], [], 0
    for name, sz in names:
        if cur and cur_sz + sz > P.BIN_LIMIT:
            bins.append(cur)
            cur, cur_sz = [], 0
        cur.append((name, sz))
        cur_sz += sz
    if cur:
        bins.append(cur)

    problems = 0
    for n, files in enumerate(bins, 1):
        ds_id = f"{P.OWNER}/{P.PREFIX}-{n:02d}"
        got = kaggle_files(ds_id)
        missing = [os.path.basename(x) for x, _ in files if os.path.basename(x) not in got]
        sizediff = [(os.path.basename(x), shards[x] / 1e9, got.get(os.path.basename(x), 0))
                    for x, _ in files
                    if os.path.basename(x) in got
                    and abs((got[os.path.basename(x)] or 0) - shards[x] / 1e9) > 0.05]
        priv = flags.get(ds_id, "missing")
        status = "OK"
        if missing:
            status, problems = f"MISSING {missing}", problems + 1
        elif sizediff:
            status, problems = f"SIZE MISMATCH {sizediff}", problems + 1
        if priv not in (False, "False", None):
            status += f" | visibility={priv}"
            problems += 1
        print(f"{ds_id}: {status}  (files: {len(got)})")

    print(f"\n{'ALL VERIFIED — sizes match, all public' if problems == 0 else f'{problems} PROBLEMS FOUND'}")


if __name__ == "__main__":
    main()
