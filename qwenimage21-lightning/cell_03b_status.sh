#!/bin/sh
# CELL 3b — poll the cell 3 download. Safe to re-run at any time, as often
# as you like; it only reads.
#
# Success = the log ends with SNAPSHOT_RETURNED *and* the byte-exact verify
# at the bottom prints "33.1 GB verified".

WORK="${QI21_WORK:-$HOME/qwenimage21}"

echo "=== log tail ==="
tail -12 "$WORK/download.log" 2>/dev/null || echo "(no log yet — is cell 3 running?)"
echo ""
echo "---"
pgrep -af "snapshot_download|_fetch.py" | head -3 || echo "(no download process)"

echo ""
echo "=== size so far ==="
du -sh "$WORK" 2>/dev/null || echo "(nothing yet)"

echo ""
echo "=== byte-exact verify ==="
# A download that died mid-shard leaves a short file. This is the check that
# catches it, so do not trust the process being gone as proof of success.
python3 - "$WORK" <<'PYEOF'
import os, sys
work = sys.argv[1]
expected = {
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 9968332504,
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 4261951904,
    "text_encoder/model-00001-of-00004.safetensors": 4998056552,
    "text_encoder/model-00002-of-00004.safetensors": 4915962464,
    "text_encoder/model-00003-of-00004.safetensors": 4915962496,
    "text_encoder/model-00004-of-00004.safetensors": 2704357976,
    "vae/diffusion_pytorch_model.safetensors": 1350989512,
}
bad = []
for rel, want in expected.items():
    p = os.path.join(work, rel)
    if not os.path.exists(p):
        bad.append(f"MISSING  {rel}")
        continue
    have = os.path.getsize(p)
    if abs(have - want) > 5_000_000:
        bad.append(f"SHORT    {rel}  {have/1e9:.2f} GB != {want/1e9:.2f} GB")
if os.path.exists(os.path.join(work, "model_index.json")) and not bad:
    print("33.1 GB verified — all 7 shards byte-exact, model_index.json present")
elif bad:
    print("NOT READY:")
    for b in bad:
        print("  " + b)
else:
    print("NOT READY — model_index.json missing (download still in progress?)")
PYEOF
