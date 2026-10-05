#!/bin/sh
# CELL 1 — machine report. Free, no installs, no GPU touched beyond one
# allocation. Run this FIRST: it prints the two facts everything downstream
# assumes (the card is Blackwell sm_120, and the disk can hold 33.1 GB).
#
# Studio: RTX PRO 6000 96GB, Interruptible OFF, 4h duration (~$3.26/h).
# Cost of THIS cell: $0 (it is seconds of work, and the Studio sleeps
# automatically after 10 min idle, so a forgotten session stops billing).

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,driver_version,persistence_mode \
  --format=csv,noheader
echo ""
echo "If compute_cap reads 12.0 you are on Blackwell and FP8 is native."
echo "If it reads 8.9 you are on Ada (still fine). Anything else: stop and"
echo "paste this output — the rest of the kit assumes a modern card."

echo ""
echo "=== RAM / DISK ==="
free -g | head -2
df -h / /teamspace "$HOME" 2>/dev/null | awk 'NR==1 || !seen[$6]++'

echo ""
echo "=== python / torch ==="
python3 -c "import sys; print('python', sys.version.split()[0])"
python3 - <<'PYEOF' 2>&1 | tail -20
import torch
print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpus       :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i}   : {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB")
    # The two dtype questions this kit cares about. Blackwell/Ada do BF16
    # natively; FP8 needs sm_89+. A T4 (sm_75) fails BOTH, which is exactly
    # why the Kaggle route in the README has to quantize to INT8 instead.
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul:", bool(torch.isfinite(a @ a).all()))
    print("fp8 dtype  :", hasattr(torch, "float8_e4m3fn"))
PYEOF

echo ""
echo "=== writable work dir ==="
# Lightning persists the Studio home (~) across restarts, which is where the
# 33.1 GB of weights must go — /tmp is RAM-ish and does not survive.
WORK="$HOME/qwenimage21"
mkdir -p "$WORK"
echo "WORK=$WORK"
df -h "$WORK" | awk 'NR==2{print "free on WORK:", $4}'
echo "33.1 GB of BF16 weights must fit above. If it does not, stop here."

echo ""
echo "CELL 1 OK — read compute_cap before going on"
