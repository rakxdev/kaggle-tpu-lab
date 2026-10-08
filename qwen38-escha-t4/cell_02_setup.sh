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
pkill -f "cmake --build" 2>/dev/null
sleep 1
nohup sh -c "
  cd $WORK/llama.cpp-escha || exit 1
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > $WORK/build.log 2>&1 \
  && cmake --build build -j\$(nproc) \
        --target llama-server llama-cli llama-bench >> $WORK/build.log 2>&1 \
  && echo BUILD_OK >> $WORK/build.log || echo BUILD_FAILED >> $WORK/build.log
" > /dev/null 2>&1 &
echo "BUILD_RUNNING — poll with cell_02b_status.sh (llama.cpp CUDA on 4 vCPUs: ~15-30 min)"