# CELL 4 — pack the Coder GGUF into PUBLIC Kaggle datasets (the Flash-Next
# pattern). The model is 59.3 GB and Kaggle's per-session scratch quota is
# 57.6 GiB (hit live 2026-10-04), so it can never fit in temp — but one ~29 GB
# shard at a time fits fine: download -> push dataset -> delete local -> next.
# After packing: attach the datasets in the sidebar (Add Input), then RE-RUN
# this cell — it detects the mounts and builds the zero-disk symlink view.
# Progress (re-run any time):  !tail -5 /kaggle/working/pack.log

import json, os, subprocess, sys, time, urllib.request

OWNER = "rakeshbehera42"
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
PREFIX = "qwen38-fn-coder"
WORK = "/kaggle/tmp/pack"
BIN_LIMIT = 30.0e9   # per-dataset budget — well under the 57.6 GiB scratch quota
GGUF_DIR = "/kaggle/tmp/gguf-coder"

TOKEN = open("/tmp/hf_token").read().strip()
os.environ["HF_TOKEN"] = TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"   # the fast lane (hf_transfer is deprecated)

# ---- already mounted? then: symlink farm + verify (fast path) ------------
mounted = {}
if os.path.isdir("/kaggle/input"):
    for d in sorted(os.listdir("/kaggle/input")):
        if d.startswith(PREFIX):
            for f in os.listdir(f"/kaggle/input/{d}"):
                p = f"/kaggle/input/{d}/{f}"
                if os.path.isfile(p):
                    mounted[f] = p

req = urllib.request.Request(f"https://huggingface.co/api/models/{REPO}?blobs=true")
with urllib.request.urlopen(req, timeout=60) as r:
    api = json.load(r)
files = {s["rfilename"]: (s.get("size") or 0) for s in api["siblings"]
         if (s.get("size") or 0) > 0 and not s["rfilename"].startswith(".")}
total_gb = sum(files.values()) / 1e9

if mounted:
    have = {f: os.path.getsize(p) for f, p in mounted.items()}
    missing = {f: s for f, s in files.items()
               if f not in have or abs(have.get(f, 0) - s) > 1e6}
    os.makedirs(GGUF_DIR, exist_ok=True)
    for f, p in mounted.items():
        dst = f"{GGUF_DIR}/{f}"
        if not os.path.exists(dst):
            os.symlink(p, dst)
    print(f"datasets mounted: {len(mounted)} files, {sum(have.values())/1e9:.1f} GB")
    if missing:
        print(f"!! MISSING vs HF manifest ({len(missing)}): {sorted(missing)[:5]}")
        print("   attach every dataset the packer printed, then re-run this cell")
    else:
        print(f"CELL 4 OK — {total_gb:.1f} GB complete via datasets, view at {GGUF_DIR}")
    sys.exit(0)

# ---- not mounted: run the packer (detached, ~1-2 h) -----------------------
auth = subprocess.run("kaggle datasets list --mine --page-size 1", shell=True,
                      capture_output=True, text=True)
if auth.returncode != 0:
    print("!! kaggle CLI not authenticated in this session:", auth.stderr[-300:])
    sys.exit(1)

packer = '''
import json, os, shutil, subprocess, time, urllib.request
REPO, OWNER, PREFIX, WORK, BIN_LIMIT = %(repo)s, %(owner)s, %(prefix)s, %(work)s, %(limit)s
def log(m): print(f"[{time.strftime('%%H:%%M:%%S')}] {m}", flush=True)
req = urllib.request.Request(f"https://huggingface.co/api/models/{REPO}?blobs=true")
with urllib.request.urlopen(req, timeout=60) as r:
    api = json.load(r)
files = {s["rfilename"]: (s.get("size") or 0) for s in api["siblings"]
         if (s.get("size") or 0) > 0 and not s["rfilename"].startswith(".")}
items = sorted(files.items(), key=lambda t: (t[1], t[0]), reverse=True)
bins, cur, cur_sz = [], [], 0
for name, size in items:
    if cur and cur_sz + size > BIN_LIMIT:
        bins.append(cur); cur, cur_sz = [], 0
    cur.append(name); cur_sz += size
if cur: bins.append(cur)
log(f"manifest: {len(files)} files {sum(files.values())/1e9:.1f} GB -> {len(bins)} datasets")
for i, names in enumerate(bins, 1):
    ds_id = f"{OWNER}/{PREFIX}-{i:02d}"
    folder = f"{WORK}/{PREFIX}-{i:02d}"
    os.makedirs(folder, exist_ok=True)
    for name in names:
        dest = os.path.join(folder, os.path.basename(name))
        if os.path.exists(dest) and abs(os.path.getsize(dest) - files[name]) < 1e6:
            log(f"  cached: {name}"); continue
        url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
        t0 = time.time(); tmp = dest + ".part"
        log(f"  downloading {name} ({files[name]/1e9:.1f} GB)...")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(16 << 20)
                if not b: break
                f.write(b)
        os.replace(tmp, dest)
        log(f"  done {name} in {(time.time()-t0)/60:.1f} min")
    meta = {"id": ds_id, "title": ds_id.split("/")[1], "licenses": [{"name": "other"}]}
    json.dump(meta, open(os.path.join(folder, "dataset-metadata.json"), "w"))
    subprocess.run(f"kaggle datasets delete -y {ds_id}", shell=True, capture_output=True)
    for attempt in (1, 2):
        r = subprocess.run(f"kaggle datasets create -p {folder} -u", shell=True,
                           capture_output=True, text=True)
        if r.returncode == 0: break
        log(f"  create attempt {attempt} failed: {(r.stdout + r.stderr)[-300:]}")
        time.sleep(20)
    if r.returncode != 0:
        log(f"!! dataset {ds_id} failed - stopping"); raise SystemExit(1)
    log(f"DATASET {ds_id} UPLOADED ({sum(files[n] for n in names)/1e9:.1f} GB)")
    shutil.rmtree(folder, ignore_errors=True)
log("PACKING COMPLETE - attach these in the sidebar (Add Input):")
for i in range(1, len(bins) + 1):
    log(f"  {OWNER}/{PREFIX}-{i:02d}")
''' % {"repo": repr(REPO), "owner": repr(OWNER), "prefix": repr(PREFIX),
       "work": repr(WORK), "limit": repr(BIN_LIMIT)}
open("/kaggle/working/pack_coder.py", "w").write(packer)
subprocess.run(f"rm -rf {WORK} {GGUF_DIR}", shell=True)
subprocess.Popen("python3 /kaggle/working/pack_coder.py > /kaggle/working/pack.log 2>&1",
                 shell=True)
print("PACKER LAUNCHED — poll with:  !tail -5 /kaggle/working/pack.log")
print("When it says PACKING COMPLETE: attach the listed datasets in the")
print("sidebar (Add Input), then RE-RUN this cell for the symlink view.")
