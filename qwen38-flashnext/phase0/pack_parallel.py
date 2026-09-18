#!/usr/bin/env python3
"""Parallel packer: download all remaining shards concurrently, then run
3 concurrent kaggle dataset uploads. Viable since staging moved to
/kaggle/tmp (~1 TB) — the 20 GB /kaggle/working cap no longer forces
serialization. Skips datasets already complete."""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pack_datasets as P


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_bins():
    shards = P.shard_sizes()
    bins, cur, cur_sz = [], [], 0
    for name, sz in shards:
        if cur and cur_sz + sz > P.BIN_LIMIT:
            bins.append(cur)
            cur, cur_sz = [], 0
        cur.append((name, sz))
        cur_sz += sz
    if cur:
        bins.append(cur)
    return bins


def main():
    os.makedirs(P.WORK, exist_ok=True)
    bins = compute_bins()
    todo = []
    for n, files in enumerate(bins, 1):
        ds_id = f"{P.OWNER}/{P.PREFIX}-{n:02d}"
        if P.dataset_complete(ds_id, files):
            log(f"dataset {n:02d} already complete — skipped")
            continue
        folder = os.path.join(P.WORK, f"chunk{n:02d}")
        os.makedirs(folder, exist_ok=True)
        sizes = dict(files)
        targets = []
        for name, _ in files:
            dest = os.path.join(folder, os.path.basename(name))
            if not (os.path.exists(dest) and abs(os.path.getsize(dest) - sizes[name]) < 1e6):
                targets.append((name, dest))
        todo.append((n, ds_id, folder, targets, len(files)))
    log(f"{len(todo)} datasets to pack: {', '.join(str(t[0]) for t in todo)}")

    # ---- phase 1: download every remaining shard, heavily parallel ----
    all_targets = [(n, name, dest) for n, _, _, targets, _ in todo for name, dest in targets]
    log(f"phase 1: downloading {len(all_targets)} shards concurrently")
    t0 = time.time()
    done_bytes = [0]

    def dl(job):
        n, name, dest = job
        P.download(name, dest)
        return os.path.getsize(dest)

    with ThreadPoolExecutor(max_workers=10) as ex:
        for sz in ex.map(dl, all_targets):
            done_bytes[0] += sz
    dt = time.time() - t0
    log(f"phase 1 done: {done_bytes[0]/1e9:.1f} GB in {dt/60:.1f} min "
        f"({done_bytes[0]/1e6/max(dt,1):.0f} MB/s aggregate)")

    # ---- phase 2: concurrent uploads ----
    log("phase 2: creating datasets (3 concurrent uploads)")

    def upload(job):
        n, ds_id, folder, targets, nfiles = job
        t = time.time()
        try:
            res = P.create_dataset(folder, ds_id)
        except Exception as e:
            res = f"error: {e!r}"[:250]
        P.sh(f"rm -rf {folder}", check=False)
        log(f"dataset {n:02d} -> {res} ({(time.time()-t)/60:.1f} min)")
        return res

    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(upload, job): job[0] for job in todo}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    log("=== SUMMARY ===")
    ok = sum(1 for r in results.values() if r in ("created-public", "skipped-complete"))
    for n in sorted(results):
        log(f"  dataset {n:02d}: {results[n]}")
    log(f"{ok}/{len(bins)} datasets ready overall")


if __name__ == "__main__":
    main()
