"""Flash-Next IQ3_S endpoint on Modal — A100-80GB, llama.cpp server, OpenAI API.

Design (all source-cited):
- Weights: ISTA-DASLab GSQ-RCO IQ3_S — 54.8 GB weights shard resident in VRAM,
  28.8 GB n-gram shard mmap'd (page cache = RAM), per the model card's
  "Memory requirements" table. Fits 80 GB with ~20 GB left for KV + MTP + mmproj.
- Engine: official llama.cpp CUDA server image, pinned to release b11379
  (qwen4exp day-0 support; GSQ-RCO quants load in vanilla llama.cpp).
- Serving: @app.server (Modal Servers) with a public URL, scale-to-zero
  (min_containers=0), llama-server --api-key as the only auth gate.

The API key is NOT stored in this file: deploy with APP_API_KEY set in the
environment (cell 4 sets it in the notebook) and it is injected via `env=`.
"""

import os
import socket
import subprocess
import time
import urllib.request

import modal

MINUTES = 60  # seconds
PORT = 8000
CACHE = "/cache"
VOL = modal.Volume.from_name("flashnext-weights", create_if_missing=True)

MODEL = f"{CACHE}/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
MMPROJ = f"{CACHE}/gguf/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
MTP = f"{CACHE}/gguf/mtp/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"

# Deploy-time requirement: run with APP_API_KEY set (cell 4a does this).
API_KEY = os.environ["APP_API_KEY"]

image = modal.Image.from_registry(
    "ghcr.io/ggml-org/llama.cpp:server-cuda-b11379", add_python="3.12"
).entrypoint([])  # image entrypoint is llama-server itself; clear it for Modal

app = modal.App("flashnext-endpoint", image=image)


def wait_ready(proc):
    """Block until llama-server answers /health with 200 (503 while loading)."""
    while True:
        if proc.poll() is not None:  # fail fast if the engine died
            raise RuntimeError(f"llama-server exited with code {proc.poll()}")
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=5).close()
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=30).close()
            return
        except Exception:
            time.sleep(3)


@app.server(
    volumes={CACHE: VOL},
    gpu="A100-80GB",          # $2.50/h per-second billing, scales to zero
    cpu=4,
    memory=49152,             # 48 GiB — page-cache room for the 28.8 GB n-gram shard
    port=PORT,
    startup_timeout=20 * MINUTES,   # first load: 55 GB Volume -> VRAM + mmap warmup
    scaledown_window=15 * MINUTES,  # keep warm between requests; lower to save credit
    min_containers=0,               # 0 = scale to zero when idle
    exit_grace_period=30,
    unauthenticated=True,           # URL is public; --api-key below is the real gate
    env={"APP_API_KEY": API_KEY},   # injected into the container for the server process
)
class FlashNext:
    @modal.enter()
    def start(self):
        missing = [p for p in (MODEL, MMPROJ, MTP) if not os.path.exists(p)]
        if missing:
            import glob

            found = glob.glob(f"{CACHE}/gguf/**/*.gguf", recursive=True)
            raise RuntimeError(f"missing {missing}; found on volume: {found}")

        cmd = [
            "/app/llama-server",
            "--model", MODEL,
            "--mmproj", MMPROJ,
            "-md", MTP,                                    # MTP draft head
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
            "-ngl", "99",                                  # everything resident: 54.8 GB << 80 GB
            "-lm", "mmap", "--lazy-mode", "on",            # n-gram shard: mmap, paged into RAM
            "-fa", "on",
            "-ctk", "q8_0", "-ctv", "q8_0",                # half-size KV cache
            "-c", "32768",                                 # raise after the smoke benchmark
            "-b", "2048", "-ub", "512",
            "--jinja", "--alias", "flashnext-iq3s",
            "--api-key", os.environ["APP_API_KEY"],
            "--host", "0.0.0.0", "--port", str(PORT),
        ]
        print("starting llama-server:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, start_new_session=True)
        wait_ready(self.proc)

    @modal.exit()
    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
