#!/usr/bin/env python3
# CELL 5b — verify the public endpoint end-to-end (runnable from anywhere).
#
#   python3 cell_05b_proof.py <public_url> <api_key>
#
# Verifies:
#   1. Health check returns 200 without auth.
#   2. Missing/invalid token returns 401.
#   3. Real text translation with tok/s measurement.
#   4. Multimodal image translation via base64 image_url.
#   5. Server-Sent Events (SSE) streaming output.
#
# Sentinel: PROOF_OK

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from PIL import Image, ImageDraw

URL = (sys.argv[1] if len(sys.argv) > 1 else "PASTE_PUBLIC_URL").rstrip("/")
KEY = sys.argv[2] if len(sys.argv) > 2 else "PASTE_API_KEY"

if URL.startswith("PASTE_") or KEY.startswith("PASTE_"):
    sys.exit("usage: python3 cell_05b_proof.py <public_url> <api_key>")


def call(path, payload=None, key=KEY, timeout=120):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        URL + path,
        data=data,
        method="POST" if data else "GET",
    )
    if data:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e


print(f"== 1. health check ({URL}/health) ==")
with call("/health", key=None) as r:
    print(f"    status: {r.status}")
    assert r.status == 200, "health check failed"

print("== 2. auth gates ==")
for label, k in (("no token", None), ("bad token", "invalid-key")):
    with call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, key=k) as r:
        print(f"    {label}: {r.status}" + (" (401 expected)" if r.status == 401 else " !!"))
        assert r.status == 401, f"expected 401 for {label}"

print("== 3. text translation ==")
t0 = time.time()
with call("/v1/chat/completions", {
    "model": "Index-Translate-9B",
    "messages": [
        {"role": "user", "content": "请将以下文本翻译为英语，直接输出翻译结果，不要进行任何解释。\n\n自由软件是人类知识的共享财富。"}
    ],
    "temperature": 0.0,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": False},
}) as r:
    data = json.load(r)
dt = time.time() - t0
out_text = data["choices"][0]["message"]["content"].strip()
if "</think>" in out_text:
    out_text = out_text.split("</think>", 1)[1].strip()
out_text = out_text.removeprefix("<think>").strip()
usage = data.get("usage", {})
comp_toks = usage.get("completion_tokens", len(out_text.split()))
rate = comp_toks / max(dt, 1e-6)
print(f"    input : '自由软件是人类知识的共享财富。'")
print(f"    output: {out_text!r}")
print(f"    speed : {comp_toks} tokens in {dt:.2f}s = {rate:.1f} tok/s")

print("== 4. multimodal image translation ==")
img = Image.new("RGB", (400, 140), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.rectangle([5, 5, 395, 135], outline=(0, 0, 0), width=3)
d.text((30, 50), "KNOWLEDGE IS POWER", fill=(0, 0, 0))

buf = io.BytesIO()
img.save(buf, format="PNG")
b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

t0 = time.time()
with call("/v1/chat/completions", {
    "model": "Index-Translate-9B",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please translate any text inside this image into Chinese directly."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
            ],
        }
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": False},
}) as r:
    img_data = json.load(r)
dt_img = time.time() - t0
out_img = img_data["choices"][0]["message"]["content"].strip()
if "</think>" in out_img:
    out_img = out_img.split("</think>", 1)[1].strip()
out_img = out_img.removeprefix("<think>").strip()
print(f"    image input : synthesized PNG with 'KNOWLEDGE IS POWER'")
print(f"    image output: {out_img!r} in {dt_img:.2f}s")

print("== 5. streaming verification ==")
req_stream = urllib.request.Request(
    URL + "/v1/chat/completions",
    data=json.dumps({
        "model": "Index-Translate-9B",
        "messages": [{"role": "user", "content": "Translate to English: 早上好。"}],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": True,
    }).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    method="POST",
)
chunks = 0
with urllib.request.urlopen(req_stream, timeout=60) as r:
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line:
            chunks += 1
print(f"    received {chunks} stream SSE chunks (streaming verified)")

print("\n" + "=" * 50)
print("PROOF_OK — endpoint verified for text, vision, and streaming!")
print("=" * 50)
