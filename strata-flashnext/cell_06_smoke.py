# CELL 6 — smoke test the local server: health, model list, one chat
# completion with token counting (the speed first look).
import json, time, urllib.request

BASE = "http://127.0.0.1:8080"

def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.loads(r.read())

def chat(prompt, max_tokens=64):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    u = out.get("usage", {})
    text = out["choices"][0]["message"]["content"]
    return text, u, dt

print("HEALTH:", get("/health"))
print("MODELS:", [m.get("id") for m in get("/v1/models")["data"]])

text, u, dt = chat("The capital of Germany is")
print(f"\nPROOF: {text[:120]!r}")
print("usage:", u)

n = u.get("completion_tokens", 0)
if n and dt > 0:
    print(f"DECODE: ~{n / dt:.1f} tok/s (incl. prefill, {n} tokens in {dt:.1f}s)")
print("CELL 6 OK" if text else "!! empty answer")
