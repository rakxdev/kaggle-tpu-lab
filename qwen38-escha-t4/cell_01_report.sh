#!/bin/sh
# CELL 1 — machine report. Everything downstream keys off what this prints.
#
# Cost: $0 (seconds of CPU). Confirms: 2× T4 visible, CUDA toolkit + nvcc
# version (the fork builds with the image's toolkit — Kaggle T4 image is a
# CUDA 12.8 host), disk headroom for 13.2 GB of GGUFs + a llama.cpp build.
#
# Model this kit serves: aj9o9/Qwen3.8-27B-Escha-W2-GGUF — a NATIVE 2-bit
# dense 27B (2.469 bpw) that ONLY runs on the author's llama.cpp fork
# (Ajay9o9/llama.cpp-escha, branch escha-w2-dense). Stock llama.cpp refuses
# the files. The kernel has explicit Turing (sm_75) support — verified in
# source: `#ifdef TURING_MMA_AVAILABLE` tensor-core prefill path plus a
# `cc >= GGML_CUDA_CC_TURING` runtime gate with an fp32 fallback.

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" | head -1

echo ""
echo "== nvcc (the build needs it; blank means cell 2 will fail) =="
which nvcc && nvcc --version | tail -1

echo ""
echo "== CPU / RAM =="
nproc
head -2 /proc/meminfo

echo ""
echo "== disk (need >16 GB free where the models go) =="
df -h /kaggle/tmp /kaggle/working 2>/dev/null | head -5

echo ""
echo "== compiler =="
gcc --version | head -1
cmake --version | head -1

echo ""
echo "REPORT_OK — paste this whole output if anything downstream fails"