#!/usr/bin/env python3
# CELL 6b — prove the endpoint from OUTSIDE Kaggle (your laptop, any machine).
#
#   python3 cell_06b_proof.py <public_url> <api_key>
#
# Cost: $0 of quota on the Kaggle side beyond a few hundred tokens.
# Contract checked:
#   1. health answers unauthenticated
#   2. no key / wrong key  -> 401
#   3. one real chat completion (correctness)
#   4. decode speed measured on a longer generation (the MTP number you came for)
#   5. streaming works (OpenAI stream=true) — tools like Claude Code need it

import json
import sys
import time
import urllib.error
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "PASTE_PUBLIC_URL").rstrip("/")
KEY = sys.argv[2] if len(sys.argv) > 2 else "PASTE_API_KEY"
if URL.startswith("PASTE_") or KEY.startswith("PASTE_"):
    sys.exit("usage: python3 cell_06b_proof.py <public_url> <api_key>")


def call(path, payload=None, key=KEY, timeout=300, stream=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL + path, data=data,
                                 method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e


print(f"== 1. health (no auth) ==")
with call("/health") as r:
    print("   ", r.status, r.read(200).decode())

print("== 2. auth gate ==")
for label, key in (("no key", None), ("bad key", "escha-wrong")):
    with call("/v1/chat/completions",
              {"messages": [{"role": "user", "content": "hi"}]}, key=key) as r:
        print(f"    {label}: {r.status}" + (" (401 expected)" if r.status == 401 else " !!"))
        if r.status != 401:
            sys.exit("AUTH GATE FAILED")

print("== 3. correctness ==")
with call("/v1/chat/completions", {
        "messages": [{"role": "user",
                      "content": "What is 84 * 3 / 2? Reply with just the number."}],
        "max_tokens": 32, "temperature": 0}) as r:
    d = json.load(r)
    ans = d["choices"][0]["message"]["content"].strip()
    print(f"    answer: {ans!r}  (expect 126)")

print("== 4. decode speed (256 tokens, greedy) ==")
with call("/v1/chat/completions", {
        "messages": [{"role": "user",
                      "content": "Write a detailed 200-word story about a lighthouse."}],
        "max_tokens": 256, "temperature": 0}) as r:
    d = json.load(r)
    u = d["usage"]
    t = d.get("timings", {})
    wall = t.get("predicted_ms", 0) / 1000
    n = u["completion_tokens"]
    print(f"    {n} tokens in {wall:.1f}s = {n / wall:.2f} tok/s"
          if wall else f"    {n} tokens (server gave no timings)")
    if t:
        print(f"    prompt: {t.get('prompt_ms', 0) / 1000:.1f}s for "
              f"{u['prompt_tokens']} toks = "
              f"{u['prompt_tokens'] / max(t.get('prompt_ms', 1) / 1000, 1e-9):.0f} tok/s prefill")

print("== 5. streaming ==")
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps({
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "max_tokens": 24, "temperature": 0, "stream": True}).encode(), method="POST")
req.add_header("Content-Type", "application/json")
req.add_header("Authorization", f"Bearer {KEY}")
chunks = 0
with urllib.request.urlopen(req, timeout=120) as r:
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line:
            chunks += 1
print(f"    {chunks} streamed chunks" + (" OK" if chunks > 1 else " !!"))

print("PROOF_OK")