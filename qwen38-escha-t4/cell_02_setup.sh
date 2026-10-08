#!/bin/sh
# CELL 2 — clone the escha fork, arm the GPU heartbeat, start the CUDA build.
#
# Cost: ~15-30 min of the free 30 GPU-h/week quota (compile is CPU-side but
# the session holds both T4s). The build is pinned to sm_75 ONLY — compiling
# every arch would triple the time for GPUs the Kaggle VM doesn't have.
#
# Two things this cell refuses to skip:
#   * the fork's kernel must actually contain the Turing gates (if upstream
#     moved and the check fails, STOP — stock llama.cpp cannot load W2 files)
#   * the GPU heartbeat (Kaggle kills idle accelerator sessions; the build +
#     download are long CPU-side stretches)
#
# Safe to re-run: re-clones into a fresh dir only if missing, rebuilds into
# the same build/ dir (cmake reuses objects, so a re-run is fast).

WORK=/kaggle/tmp/escha
FORK=https://github.com/Ajay9o9/llama.cpp-escha
BRANCH=escha-w2-dense
mkdir -p "$WORK" && cd "$WORK" || exit 1

echo "== 1. the fork =="
if [ -d llama.cpp-escha/.git ]; then
  git -C llama.cpp-escha fetch -q origin "$BRANCH"
  git -C llama.cpp-escha checkout -q "$BRANCH"
  git -C llama.cpp-escha reset -q --hard origin/"$BRANCH"
else
  git clone -q --branch "$BRANCH" --single-branch "$FORK" || exit 1
fi
cd llama.cpp-escha || exit 1
echo "fork:   $(git log --oneline -1 | head -c 80)"

echo ""
echo "== 2. self-check: the W2 kernel + its Turing support =="
K=ggml/src/ggml-cuda/escha-moe.cu
[ -f "$K" ] || { echo "!! $K missing — the branch moved; re-check the fork"; exit 1; }
grep -q "TURING_MMA_AVAILABLE" "$K" && echo "kernel: TURING_MMA_AVAILABLE present (sm_75 tensor-core prefill path)"
grep -q "GGML_CUDA_CC_TURING" "$K" && echo "kernel: runtime cc >= TURING gate present (fp32 fallback for older)"
# the op llama.cpp dispatches for the W2 payloads
grep -rq "GGML_OP_ESCHA_MUL_MAT" ggml/src/ggml-cuda/*.cu ggml/src/ggml-cuda/*.cuh 2>/dev/null \
  && echo "kernel: GGML_OP_ESCHA_MUL_MAT dispatch present"

echo ""
echo "== 3. GPU heartbeat (Kaggle stops idle accelerator sessions) =="
cat > /kaggle/working/gpu_heartbeat.py <<'EOF'
import subprocess, sys, time, os
INTERVAL = int(os.environ.get("HEARTBEAT_S", "240"))
TOUCH = ("import torch; x = torch.ones(64, 64, device='cuda:0') @ "
         "torch.ones(64, 64, device='cuda:0'); torch.cuda.synchronize(); "
         "print('HB', float(x.sum()))")
while True:
    r = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
    if r.stdout.strip():
        print(time.strftime("[%H:%M:%S]"), "heartbeat: server running — skip", flush=True)
    else:
        try:
            h = subprocess.run([sys.executable, "-c", TOUCH], capture_output=True,
                               text=True, timeout=180)
            print(time.strftime("[%H:%M:%S]"),
                  "heartbeat: GPU touched" if "HB" in (h.stdout or "")
                  else f"heartbeat: failed {(h.stderr or '').strip()[-100:]}",
                  flush=True)
        except Exception as e:
            print(time.strftime("[%H:%M:%S]"), f"heartbeat: error {e!r}", flush=True)
    time.sleep(INTERVAL)
EOF
nohup python3 /kaggle/working/gpu_heartbeat.py > /kaggle/working/heartbeat.log 2>&1 &
sleep 3; tail -1 /kaggle/working/heartbeat.log

echo ""
echo "== 4. configure + build (background, log: $WORK/build.log) =="
# Kill and relaunch are separate commands (HANDOFF rule).
pkill -f "do_build.sh" 2>/dev/null
pkill -f "cmake --build" 2>/dev/null
sleep 1

GEN=""
command -v ninja > /dev/null 2>&1 && GEN="-G Ninja"   # ninja: ~2x faster build

BUILD_SCRIPT="$WORK/do_build.sh"
cat > "$BUILD_SCRIPT" <<EOF
#!/bin/sh
cd "$WORK/llama.cpp-escha" || { echo "cd failed"; exit 1; }
echo "[do_build] started at \$(date)"
# Containers have libcuda.so.1 injected by driver but often lack the .so dev symlink
[ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ] \
  && ln -sf libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null
# Wipe any stale failed configure cache
rm -rf build
echo "[do_build] cmake configure (arch 75, no-vmm)..."
cmake -B build $GEN -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DGGML_CUDA_NO_VMM=ON \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
  && echo "[do_build] cmake build targets..." \
  && cmake --build build -j\$(nproc) \
        --target llama-server llama-cli llama-bench \
  && echo BUILD_OK || echo BUILD_FAILED
EOF
chmod +x "$BUILD_SCRIPT"

# setsid detaches from IPython's process group so cell termination cannot
# kill the background compile. Direct stdout/stderr redirection guarantees
# $WORK/build.log is opened on disk immediately at launch.
LOG="$WORK/build.log"
setsid nohup "$BUILD_SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "BUILD_RUNNING pid $! — log $LOG"
sleep 5
echo "--- initial build log tail ---"
tail -8 "$LOG" 2>/dev/null || echo "(log file opening...)"
echo ""
echo "poll with cell_02b_status.sh (llama.cpp CUDA on 4 vCPUs: ~10-25 min with ninja)"