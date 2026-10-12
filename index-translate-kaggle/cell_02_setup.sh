#!/bin/sh
# CELL 2 — obtain llama-server with CUDA support and arm the GPU heartbeat.
#
# Fast path: downloads official prebuilt release binary for Ubuntu CUDA 12.8
# (takes ~15 seconds, zero compile time).
# Fallback: compiles from source with Ninja for SM75 (-DGGML_CUDA_NO_VMM=ON)
# in the background if the prebuilt binary cannot be used.

WORK=/kaggle/tmp/translate
BIN_DIR="$WORK/llama-bin"
mkdir -p "$WORK" "$BIN_DIR"
cd "$WORK" || exit 1

echo "== 1. GPU heartbeat (prevents Kaggle idle timeout) =="
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
sleep 2; tail -1 /kaggle/working/heartbeat.log

echo ""
echo "== 2. checking for llama-server =="
if [ -x "$BIN_DIR/llama-server" ] && "$BIN_DIR/llama-server" --version >/dev/null 2>&1; then
  echo "llama-server already present and working at $BIN_DIR/llama-server"
  echo "BUILD_OK" > "$WORK/build.log"
  echo "BUILD_OK"
  exit 0
fi

# Try fast prebuilt binary first
echo "attempting fast prebuilt release download..."
TARBALL_URL="https://github.com/ggml-org/llama.cpp/releases/download/b11574/llama-b11574-bin-ubuntu-cuda-12.8-x64.tar.gz"
if curl -sSL -m 90 "$TARBALL_URL" -o /tmp/llama_prebuilt.tar.gz 2>/dev/null; then
  mkdir -p "$BIN_DIR"
  tar -xzf /tmp/llama_prebuilt.tar.gz -C "$BIN_DIR" --strip-components=1 2>/dev/null \
    || tar -xzf /tmp/llama_prebuilt.tar.gz -C "$BIN_DIR" 2>/dev/null
  rm -f /tmp/llama_prebuilt.tar.gz
  # Check if binary executes
  if [ -x "$BIN_DIR/bin/llama-server" ]; then
    cp "$BIN_DIR/bin/llama-server" "$BIN_DIR/llama-server"
  fi
  if [ -x "$BIN_DIR/llama-server" ]; then
    export LD_LIBRARY_PATH="$BIN_DIR:$LD_LIBRARY_PATH"
    if "$BIN_DIR/llama-server" --version >/dev/null 2>&1; then
      echo "prebuilt llama-server verified successfully!"
      echo "BUILD_OK" > "$WORK/build.log"
      echo "BUILD_OK"
      exit 0
    fi
  fi
  echo "prebuilt binary needs compilation on this host — falling back to source build..."
fi

# Fallback: build from source with Ninja
echo "== 3. building from source (background, log: $WORK/build.log) =="
if [ -d "$WORK/llama.cpp/.git" ]; then
  git -C "$WORK/llama.cpp" fetch -q origin
  git -C "$WORK/llama.cpp" checkout -q b11574
else
  git clone -q --branch b11574 --depth 1 https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp" || exit 1
fi

BUILD_SCRIPT="$WORK/do_build.sh"
cat > "$BUILD_SCRIPT" <<EOF
#!/bin/sh
cd "$WORK/llama.cpp" || exit 1
echo "[do_build] started at \$(date)"
rm -rf build
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DGGML_CUDA_NO_VMM=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j\$(nproc) --target llama-server \
  && cp build/bin/llama-server "$BIN_DIR/llama-server" \
  && echo BUILD_OK || echo BUILD_FAILED
EOF
chmod +x "$BUILD_SCRIPT"

LOG="$WORK/build.log"
pkill -f "do_build.sh" 2>/dev/null
pkill -f "ninja.*llama-server" 2>/dev/null
sleep 1
setsid nohup "$BUILD_SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "BUILD_RUNNING pid $! — poll with cell_02b_status.sh"
sleep 4
tail -5 "$LOG" 2>/dev/null || echo "(compilation starting...)"
