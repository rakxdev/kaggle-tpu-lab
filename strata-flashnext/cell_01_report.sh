#!/bin/sh
# CELL 1 — machine report. One job: print the facts the route depends on.
# Expected: 2x Tesla T4, Driver 580+ (Strata needs >= 580), CUDA 13.0 line,
# ~31 GB RAM, >900 GB free on /, python3.12 with venv, AVX2 present.
echo "== GPU =="; nvidia-smi | grep -E "Tesla|CUDA Version" || nvidia-smi
echo "== RAM =="; free -g | head -2
echo "== DISK =="; df -h / /kaggle/working /kaggle/tmp 2>/dev/null | tail -3
echo "== PYTHON =="; python3 --version; python3 -c "import venv, ensurepip; print('venv OK')"
echo "== CPU FLAGS =="; grep -o -m1 "avx2" /proc/cpuinfo | head -1; nproc
echo "== GIT =="; git --version
