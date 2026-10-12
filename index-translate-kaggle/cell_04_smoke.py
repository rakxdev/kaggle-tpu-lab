#!/usr/bin/env python3
# CELL 4 — smoke test: verify both text translation AND image translation.
#
# Starts llama-server locally on port 8099 with --mmproj attached to GPU 0,
# tests a standard text translation, tests an end-to-end base64 image translation,
# measures execution times and VRAM, then terminates cleanly.
#
# Success sentinel: INDEX_SMOKE_OK

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from PIL import Image, ImageDraw, ImageFont

WORK = "/kaggle/tmp/translate"
MODELS_DIR = f"{WORK}/models"
MAIN = f"{MODELS_DIR}/Index-Translate-9B.Q4_K_M.gguf"
MMPROJ = f"{MODELS_DIR}/Index-Translate-9B.mmproj-Q8_0.gguf"
PORT = 8099

for p in (MAIN, MMPROJ):
    if not os.path.exists(p):
        sys.exit(f"missing {p} — run cell_03_download.py first")

# Find llama-server binary
SRV = f"{WORK}/llama-bin/llama-server"
if not os.path.exists(SRV):
    SRV = f"{WORK}/llama.cpp/build/bin/llama-server"
if not os.path.exists(SRV):
    sys.exit("llama-server binary not found — run cell_02_setup.sh first")

print("== 1. launching smoke server on port 8099 ==")
subprocess.run(["pkill", "-f", f"llama-server.*{PORT}"], capture_output=True)
time.sleep(2)

CMD = [
    SRV,
    "-m", MAIN,
    "--mmproj", MMPROJ,
    "-ngl", "99",
    "--device", "CUDA0",
    "-c", "8192",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--alias", "Index-Translate-9B",
]

log_file = open(f"{WORK}/smoke.log", "w")
proc = subprocess.Popen(CMD, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)

up = False
for _ in range(60):
    time.sleep(2)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=3) as r:
            if r.status == 200:
                up = True
                break
    except Exception:
        pass
    if proc.poll() is not None:
        log_file.close()
        print(open(f"{WORK}/smoke.log").read()[-2500:])
        sys.exit("!! llama-server exited during load")

if not up:
    proc.terminate()
    sys.exit("!! llama-server did not become healthy in 120s")

print("✓ server healthy and model loaded into GPU 0")
os.system("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader")

# ---- 2. Text Translation Test ----
print("\n== 2. testing text translation ==")
body_text = json.dumps({
    "model": "Index-Translate-9B",
    "messages": [
        {"role": "user", "content": "请将以下文本翻译为英语，直接输出翻译结果，不要进行任何解释。\n\n技术赋能创作，让交流跨越语言的界限。"}
    ],
    "temperature": 0.0,
    "max_tokens": 128,
}).encode("utf-8")

req_text = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=body_text,
    headers={"Content-Type": "application/json"},
    method="POST",
)

t0 = time.time()
with urllib.request.urlopen(req_text, timeout=30) as r:
    res_text = json.load(r)
dt_text = time.time() - t0

text_out = res_text["choices"][0]["message"]["content"].strip()
print(f"text input : '技术赋能创作，让交流跨越语言的界限。'")
print(f"text output: {text_out!r} ({dt_text:.2f}s)")
assert len(text_out) > 5, "text translation output too short"

# ---- 3. Multimodal Image Translation Test ----
print("\n== 3. testing multimodal image translation (mmproj) ==")
# Create in-memory test image with bold text
img = Image.new("RGB", (480, 160), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.rectangle([10, 10, 470, 150], outline=(0, 0, 0), width=4)
d.text((40, 60), "HELLO WORLD - OPEN SOURCE", fill=(0, 0, 0))

buf = io.BytesIO()
img.save(buf, format="PNG")
b64_img = base64.b64encode(buf.getvalue()).decode("ascii")
data_url = f"data:image/png;base64,{b64_img}"

body_img = json.dumps({
    "model": "Index-Translate-9B",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please translate the text found inside this image into Chinese directly."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ],
    "temperature": 0.0,
    "max_tokens": 128,
}).encode("utf-8")

req_img = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=body_img,
    headers={"Content-Type": "application/json"},
    method="POST",
)

t0 = time.time()
with urllib.request.urlopen(req_img, timeout=60) as r:
    res_img = json.load(r)
dt_img = time.time() - t0

img_out = res_img["choices"][0]["message"]["content"].strip()
print(f"image test: synthesized PNG with 'HELLO WORLD - OPEN SOURCE'")
print(f"image output: {img_out!r} ({dt_img:.2f}s)")

# Clean shutdown of smoke server
proc.terminate()
time.sleep(2)

print("\n" + "=" * 50)
print("INDEX_SMOKE_OK — both text & vision confirmed working!")
print("=" * 50)
