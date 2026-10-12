#!/bin/sh
# CELL 1 — machine report: GPUs, CUDA toolkit, disk, and memory inspection.
#
# Hardware target: Kaggle 2x T4 (SM75, 15.83 GiB VRAM each).
# Model: Index-Translate-9B Q4_K_M (5.78 GB) + mmproj vision encoder (0.62 GB).
# Footprint fits 100% on GPU 0 with ~9 GB VRAM headroom for large image contexts.

echo "== GPU Devices =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" | head -1

echo ""
echo "== CUDA Compiler (nvcc) =="
which nvcc >/dev/null 2>&1 && nvcc --version | tail -1 || echo "(nvcc not found; prebuilt binary will be used)"

echo ""
echo "== CPU / System RAM =="
nproc | xargs -I{} echo "CPU cores: {}"
head -2 /proc/meminfo

echo ""
echo "== Storage / Scratch Space =="
df -h /kaggle/tmp /kaggle/working / 2>/dev/null | head -5

echo ""
echo "REPORT_OK"
