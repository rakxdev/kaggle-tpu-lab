#!/bin/sh
# CELL 1 — Machine report: GPUs, CUDA toolkit, disk, and memory inspection.
#
# Hardware target: Kaggle 2x T4 (SM75, 15.83 GiB VRAM each).
# Model: IndexTTS-2.5 (~0.8B - 1.5B parameters, ~6.5 - 7.5 GB VRAM in FP16).
# Fits 100% on GPU 0 with ~8 GB VRAM headroom.

echo "== GPU Devices =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" | head -1

echo ""
echo "== PyTorch & CUDA Detection =="
python3 -c "
import torch
print('  PyTorch Version :', torch.__version__)
print('  CUDA Available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  Device Count    :', torch.cuda.device_count())
    print('  Device 0 Name   :', torch.cuda.get_device_name(0))
    print('  Device 0 Memory :', f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB')
" 2>/dev/null || echo "  (PyTorch check failed or not installed)"

echo ""
echo "== CPU / System RAM =="
nproc | xargs -I{} echo "CPU cores: {}"
head -2 /proc/meminfo

echo ""
echo "== Storage / Scratch Space =="
df -h /kaggle/tmp /kaggle/working / 2>/dev/null | head -5

echo ""
echo "REPORT_OK"
