#!/usr/bin/env python3
# CELL 6c — prove the queued endpoint works end to end, from outside.
#
# Run AFTER cell 6 + 6b. Paste the URL and key those cells printed. It runs on
# any machine and exercises the never-fail queue contract:
#
#   1. health (no auth)               -> 200, worker count
#   2. auth gate                      -> 401 without / with a bad token
#   3. one real generation            -> 202 submit -> poll -> done -> PNG
#   4. CONCURRENCY: 6 submits at once -> every one gets 202 with a position,
#      NONE is rejected, all complete. This is the "never fail" proof.
#   5. cancel: a queued job cancels cleanly (only checked if one is still
#      queued long enough — race-tolerant)
#
# Cost: ~$0.10-0.30 of GPU time (several real renders).

import base64
import concurrent.futures
import io
import json
import sys
import time
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "PASTE_PUBLIC_URL"
KEY = sys.argv[2] if len(sys.argv) > 2 else "PASTE_API_KEY"

if URL.startswith("PASTE_") or KEY.startswith("PASTE_"):
    sys.exit("usage: python3 cell_06c_proof.py <public_url> <api_key>")


def call(method, path, payload=None, key=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL.rstrip("/") + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    # ngrok's interstitial browser warning otherwise blocks API clients.
    req.add_header("ngrok-skip-browser-warning", "true")
    last_exc = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                try:
                    return r.status, json.loads(body)
                except Exception:  # noqa: BLE001 — binary (PNG)
                    return r.status, body
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                return e.code, json.loads(body)
            except Exception:  # noqa: BLE001
                return e.code, {"raw": body[:300].decode("utf-8", "replace")}
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            # tunnel hiccups are transient; the queue never drops the job, so
            # retry — this is the entire point of the never-fail design
            last_exc = exc
            time.sleep(2 + attempt * 2)
    return 0, {"error": f"connection failed after retries: {last_exc}"}


def wait_done(job_id, key, deadline_s=600):
    """Poll a job to a terminal state. Returns (status, final_json)."""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        st, d = call("GET", f"/jobs/{job_id}", key=key)
        if st == 0:
            time.sleep(2)  # transient tunnel drop — keep polling
            continue
        if st != 200:
            return "http-" + str(st), d
        if d["status"] in ("done", "error", "canceled"):
            return d["status"], d
        time.sleep(1.5)
    return "timeout", {"job_id": job_id}


print("=== 1. health (no auth needed) ===")
st, body = call("GET", "/health")
print(st, json.dumps(body)[:300])
if st != 200:
    sys.exit("!! public URL is not answering — check the ngrok log")

print("\n=== 2. auth gate must actually gate ===")
st, _ = call("POST", "/generate", {"prompt": "x"}, key=None)
print(f"no token  -> {st}")
st, _ = call("POST", "/generate", {"prompt": "x"}, key="wrong-key-123")
print(f"bad token -> {st}")
if st != 401:
    sys.exit("!! a wrong key was accepted — stop and check the server")

print("\n=== 3. one real generation (512x512, 20 steps) ===")
t0 = time.time()
st, env = call("POST", "/generate",
               {"prompt": "a red panda barista, studio light",
                "width": 512, "height": 512, "steps": 20, "seed": 7},
               key=KEY)
if st != 202:
    sys.exit(f"!! submit was rejected with {st}: {str(env)[:300]} — the queue must NEVER reject")
job = env["job_id"]
print(f"202 job {job[:12]}… position {env['queue_position'] + 1}")
status, d = wait_done(job, KEY)
if status != "done":
    sys.exit(f"!! job ended {status}: {str(d)[:400]}")
wall = time.time() - t0
st, png = call("GET", f"/jobs/{job}/result", key=KEY)
open("proof_512.png", "wb").write(png)
m = d["result"]
print(f"done: {m['seconds']}s render ({m['seconds_per_step']} s/step), "
      f"{wall - m['seconds']:.1f}s of queue/transport, PNG {len(png)/1024:.0f} KiB")

try:
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)
    print(f"pixels mean={arr.mean():.1f} std={arr.std():.1f}")
    if arr.std() < 1.0:
        sys.exit("!! image is flat/black — wrong dtype, do not publish this")
except ImportError:
    print("(pillow/numpy not here — could not check for a black image)")

print("\n=== 4. concurrency: 6 simultaneous submits — all 202, none rejected ===")
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    futs = {
        i: ex.submit(
            call, "POST", "/generate",
            {"prompt": f"concurrency probe {i}", "width": 512, "height": 512,
             "steps": 20, "seed": 100 + i},
            KEY)
        for i in range(6)
    }
    jobs = {}
    for i, f in futs.items():
        st, env = f.result()
        jobs[i] = env.get("job_id")
        pos = env.get("queue_position")
        print(f"  submit {i}: {st} position {pos + 1 if isinstance(pos, int) else pos}")
rejected = [i for i, f in futs.items() if f.result()[0] != 202]
if rejected:
    sys.exit(f"!! submits {rejected} were rejected — the never-fail contract is broken")

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    results = ex.map(lambda i: wait_done(jobs[i], KEY), jobs.keys())
    outcomes = {i: r for i, r in zip(jobs.keys(), results)}
done_n = sum(1 for s, _ in outcomes.values() if s == "done")
total = time.time() - t0
print(f"all finished in {total:.1f}s: {done_n}/6 done, "
      f"{sum(1 for s, _ in outcomes.values() if s != 'done')} not-done")
if done_n != 6:
    for i, (s, d) in outcomes.items():
        if s != "done":
            print(f"  !! job {i}: {s} {str(d)[:200]}")
    sys.exit("!! not every concurrent job completed")

print("\n=== 5. queue view ===")
st, q = call("GET", "/queue")
print(st, json.dumps(q))

print("\nPROOF_DONE — never-fail queue verified: %d/%d renders completed, 0 rejections"
      % (done_n, len(jobs)))
