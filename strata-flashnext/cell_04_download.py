# CELL 4 — verify/complete the Coder GGUF in temp. RECURSIVE this time: the
# shards live in the IQ1_M/ subdirectory, which the old top-level check
# misread as "incomplete" — the 4-minute Xet download had actually succeeded.
# If anything is missing, it resumes via snapshot_download (~5 min total).
import importlib, json, os, subprocess, sys, time, urllib.request

for _pkg in ("huggingface_hub",):
    try: importlib.import_module(_pkg)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
DEST = "/kaggle/tmp/gguf-coder"
GGUF_DIR = f"{DEST}/IQ1_M"          # shards are flat in here — what --gguf-dir wants
EXPECTED = {
    "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf": 29.61e9,
    "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf": 28.80e9,
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf": 0.91e9,
}

TOKEN = open("/tmp/hf_token").read().strip()
os.environ["HF_TOKEN"] = TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"   # fast lane (hf_transfer deprecated)

def actual_sizes():
    out = {}
    for root, _, fnames in os.walk(DEST):
        for fn in fnames:
            p = os.path.join(root, fn)
            out[os.path.relpath(p, DEST)] = os.path.getsize(p)
    return out

have = actual_sizes() if os.path.isdir(DEST) else {}
def ok(name):
    exp = EXPECTED[name]
    return name in have and abs(have[name] - exp) < 5e6

missing = [n for n in EXPECTED if not ok(n)]
if missing:
    print(f"missing/incomplete: {missing} — resuming download (~5 min)...")
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(REPO, local_dir=DEST, max_workers=8)
    print(f"resume finished in {(time.time() - t0) / 60:.1f} min")
    have = actual_sizes()
    missing = [n for n in EXPECTED if not ok(n)]
    if missing:
        print(f"!! STILL missing after resume: {missing} — paste me this output")
        sys.exit(1)

# the Xet chunk cache duplicates the payload — drop it once shards are verified
cache = f"{DEST}/.cache"
if os.path.isdir(cache):
    saved = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(cache) for f in fs)
    subprocess.run(["rm", "-rf", cache])
    print(f"freed Xet cache: {saved / 1e9:.1f} GB")

print("shards verified byte-exact:")
for n, s in EXPECTED.items():
    print(f"  {have.get(n, 0) / 1e9:6.2f} GB  {n}")
print(f"CELL 4 OK — give setup this path: {GGUF_DIR}")
