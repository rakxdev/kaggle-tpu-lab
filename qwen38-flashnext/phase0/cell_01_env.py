# Phase 0 / cell 1 — machine report (CPU session)
# What we need from this: the "MemAvailable" line (host RAM decides the
# n-gram-table strategy) and confirmation there is no GPU/TPU here.

import os
import sys
import urllib.request

print("=== CPU / memory ===")
os.system("nproc")
os.system("free -g")
os.system("df -h /kaggle/working / /tmp 2>/dev/null | head -6")

print("=== accelerators (expect none) ===")
os.system("nvidia-smi -L 2>&1 | head -2")

print("=== python ===")
print(sys.version)

print("=== internet ===")
try:
    r = urllib.request.urlopen("https://huggingface.co", timeout=15)
    print("huggingface.co:", r.status)
    r2 = urllib.request.urlopen("https://github.com", timeout=15)
    print("github.com:", r2.status)
    print("INTERNET OK")
except Exception as e:
    print("INTERNET FAIL:", e)

print("CELL 1 OK")
