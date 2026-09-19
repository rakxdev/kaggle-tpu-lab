#!/usr/bin/env python3
"""qwen38-27b-gpu / phase1 — serve the INT4 model behind a public OpenAI endpoint.

vLLM's own OpenAI server (chat completions, streaming, tool calls, embeddings
of the chat template) on :8000, an API key for auth, and a cloudflared quick
tunnel for a public URL. Reuses the tunnel retry/probe logic proven by the GLM
kernel (../glm53-flash/kernel/serve_glm53.py).

All three Phase 0 fixes are baked in via phase0/cu_env.py and the flags below:
  - cu_env.reexec() puts vllm's cu130 runtime on the loader path (Kaggle's T4
    image is a CUDA-12.8 host and vllm's _C has no rpath)
  - this module must stay import-clean at module level (vllm spawns its engine
    core, which re-imports us in the child) — everything happens in main()
  - --kv-cache-dtype float16: vllm defaults this arch to fp8, T4 is SM75

Run after Phase 0: the checkpoint must already be at /kaggle/tmp/ckpt.
"""

import json
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase0"))

# Must run before anything imports vllm/torch; keep this import at the top and
# do not construct engines at module level (vllm's spawned core re-imports us).
import cu_env  # noqa: E402

cu_env.reexec()

CKPT = Path("/kaggle/tmp/ckpt")
VLLM_LOG = Path("/kaggle/working/vllm_serve.log")
CLOUDFLARED = Path("/tmp/cloudflared")
PORT = 8000
MODEL_NAME = "qwen38-27b-int4"
CTX = int(os.environ.get("QWEN_GPU_CTX", "32768"))
KEEPALIVE_MIN = int(os.environ.get("QWEN_GPU_KEEPALIVE_MIN", "480"))
TOOL_PARSER = os.environ.get("QWEN_GPU_TOOL_PARSER", "hermes")
# If set, READY/endpoint/death events are published to this ntfy topic so the
# URL and API key reach you (or the driver) without copying cell output.
NTFY_TOPIC = os.environ.get("QWEN_GPU_NTFY_TOPIC", "")
NTFY_TOKEN = os.environ.get("QWEN_GPU_NTFY_TOKEN", "")

T0 = time.time()
API_KEY = "qwen-" + secrets.token_hex(12)


def log(*parts):
    print(time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts), flush=True)


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60} s"


def publish(phase, **extra):
    """Log always; also push to ntfy if a topic is configured. Anonymous ntfy
    publishing from a Kaggle VM's shared IP gets 429s, so a token is used when
    provided (the same one the relay cell uses)."""
    log(f"PHASE {phase}" + (f" {json.dumps(extra)}" if extra else ""))
    if not NTFY_TOPIC:
        return
    try:
        body = json.dumps({"phase": phase, **extra}).encode()
        hdr = {"Content-Type": "application/json"}
        if NTFY_TOKEN:
            hdr["Authorization"] = f"Bearer {NTFY_TOKEN}"
        # publish without a title so the whole JSON lands in the message body
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=body, headers=hdr)
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:  # noqa: BLE001
        log(f"   (ntfy publish failed: {e})")


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/5  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def preflight():
    if not (CKPT / "config.json").is_file():
        log(f"!! checkpoint not found at {CKPT}")
        log("   run phase0/run_all.py first (cell 3), then this cell")
        raise SystemExit(1)
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    if len(re.findall(r"Tesla T4", out)) < 2:
        log(f"!! expected 2 T4s, nvidia-smi shows {len(re.findall(r'T4', out))}")
        raise SystemExit(1)


def vllm_flags(with_tools):
    flags = [
        "--model", str(CKPT),
        "--served-model-name", MODEL_NAME,
        "--tensor-parallel-size", "2",
        "--max-model-len", str(CTX),
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.92",
        "--kv-cache-dtype", "float16",
        '--speculative-config', '{"method": "mtp", "num_speculative_tokens": 3}',
        "--api-key", API_KEY,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--disable-log-requests",
    ]
    if with_tools:
        flags += ["--enable-auto-tool-choice", "--tool-call-parser", TOOL_PARSER]
    return flags


def launch(with_tools):
    """Start the api_server, streaming its output to VLLM_LOG. Returns the Popen."""
    log(f"$ python -m vllm.entrypoints.openai.api_server {' '.join(vllm_flags(with_tools))}")
    log(f"   vLLM output -> {VLLM_LOG}")
    f = open(VLLM_LOG, "w", buffering=1)
    return subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
                            + vllm_flags(with_tools), stdout=f, stderr=subprocess.STDOUT)


def wait_for_server(proc, wait_s=2400):
    """Poll /v1/models until the server answers; return True if it came up.
    If the child dies, print the tail of its log so the root cause is visible."""
    url = f"http://127.0.0.1:{PORT}/v1/models"
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            log(f"!! vLLM exited early with rc={proc.returncode}; last lines of {VLLM_LOG}:")
            try:
                log("   " + "\n   ".join(VLLM_LOG.read_text(errors="replace").splitlines()[-25:]))
            except Exception:  # noqa: BLE001
                pass
            return False
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
            with urllib.request.urlopen(req, timeout=15) as r:
                models = [m["id"] for m in json.loads(r.read())["data"]]
            log(f"   server up after {time.time() - t0:.0f}s, serving {models}")
            return True
        except Exception:
            time.sleep(10)
    log(f"!! no server on :{PORT} within {wait_s}s")
    return False


def start_tunnel(attempts=3, wait_s=90):
    """A cloudflared quick tunnel -> its public URL, or None. Registration
    sometimes fails or hangs: an attempt that prints no URL within wait_s is
    killed and retried. On the TPU image cloudflared segfaulted (rc -11); the
    GPU image is different, but the retry + probe covers a bad draw."""
    if not CLOUDFLARED.exists():
        log(f"   downloading cloudflared -> {CLOUDFLARED}")
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for i in range(attempts):
        tun = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                                "--no-autoupdate"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        q = queue.Queue()
        threading.Thread(target=lambda: [q.put(l) for l in iter(tun.stdout.readline, "")], daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                line = q.get(timeout=5)
            except queue.Empty:
                if tun.poll() is not None:
                    break
                continue
            m = pat.search(line)
            if m:
                log(f"   tunnel opened: {m.group(0)}")
                return m.group(0), tun
        tun.kill()
        log(f"   tunnel attempt {i + 1}/{attempts}: no URL within {wait_s}s (cloudflared rc {tun.poll()}); retrying")
    return None, None


def tunnel_probe(url, tun_holder, wait_s=180):
    """Background: confirm the public URL answers (a fresh hostname can take a
    minute to resolve); if it never does, open a new tunnel once."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=10).read()
            log(f"   reachable from outside: {url} ({time.time() - t0:.0f}s after it opened)")
            return
        except Exception:  # noqa: BLE001
            time.sleep(10)
    log(f"   tunnel URL {url} did not answer in {wait_s}s: opening a new one")
    if tun_holder[0] is not None:
        tun_holder[0].kill()
    new, tun = start_tunnel()
    if new:
        tun_holder[0] = tun
        publish("tunnel-url", endpoint=new, note="replaced-unreachable")
        log(f"#  NEW ENDPOINT: {new}   (the API key is unchanged)")
        return new
    publish("tunnel-failed", note=f"server still live inside the kernel on :{PORT}")
    log("#  tunnel gave no reachable URL; the server is still live inside the kernel on :8000")
    return None


def self_test():
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Say hi in five words."}],
            "max_tokens": 48, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr, timeout=600)
    r = json.loads(urllib.request.urlopen(req).read())
    log("   self-test:", json.dumps(r["choices"][0]["message"]["content"])[:120], r["usage"])


def main():
    banner(1, "Preflight", "checkpoint + 2 T4s")
    preflight()
    log(f"   checkpoint: {CKPT}  (ctx {CTX}, keepalive {KEEPALIVE_MIN} min)")

    banner(2, "Starting vLLM", "OpenAI server on :8000, TP=2, MTP k=3")
    with_tools = True
    proc = launch(with_tools)
    up = wait_for_server(proc)
    if not up and with_tools:
        # A wrong tool-call parser name is rejected at startup, before weights
        # load, so this fallback costs ~30 s rather than a full load. Tool
        # calls then arrive as plain text instead of structured objects.
        log("   retrying without --enable-auto-tool-choice (tool calls will come back as text)")
        with_tools = False
        proc = launch(with_tools)
        up = wait_for_server(proc)
    if not up:
        publish("failed", step="vllm-start")
        raise SystemExit(1)
    log(f"   structured tool calls: {'ON (' + TOOL_PARSER + ')' if with_tools else 'OFF (text only)'}")

    banner(3, "Self-test")
    try:
        self_test()
    except Exception as e:  # noqa: BLE001
        log(f"   self-test failed: {e!r}")

    banner(4, "Tunnel", "public cloudflared URL")
    url, tun = start_tunnel()
    tun_holder = [tun]
    if url:
        publish("tunnel-url", endpoint=url)
        threading.Thread(target=tunnel_probe, args=(url, tun_holder), daemon=True).start()
    else:
        publish("tunnel-failed", note=f"server reachable inside the kernel on :{PORT}")
        log("   no tunnel; the server is reachable inside the kernel on :8000")
    endpoint = url or f"http://127.0.0.1:{PORT}"

    banner(5, "Ready")
    log("")
    log("#" * 70)
    log(f"#  READY — the endpoint is live ({elapsed()} after start)")
    log(f"#  ENDPOINT : {endpoint}/v1   (OpenAI-compatible)")
    log(f"#  API KEY  : {API_KEY}")
    log(f"#  MODEL    : {MODEL_NAME}   (context {CTX})")
    log("#" * 70)
    log("#  Any OpenAI-compatible tool (Codex CLI, aider, opencode, curl, LangChain):")
    log(f"#    export OPENAI_BASE_URL={endpoint}/v1")
    log(f"#    export OPENAI_API_KEY={API_KEY}")
    log(f"#    export OPENAI_MODEL={MODEL_NAME}")
    log("#  curl sanity check:")
    log(f"#    curl -s {endpoint}/v1/models -H 'Authorization: Bearer {API_KEY}'")
    log("#  Claude Code (speaks OpenAI-compatible providers):")
    log(f"#    ANTHROPIC_BASE_URL is NOT used here — this endpoint is OpenAI-format only")
    log(f"#    (no /v1/messages; the GLM kernel is the one that speaks both)")
    log(f"#  Serving for up to {KEEPALIVE_MIN} min, then this cell exits on its own.")
    log(f"#  Kaggle ends GPU sessions after 12 h. Each run gets a new tunnel URL.")
    log("#" * 70)
    publish("ready", endpoint=endpoint, api_key=API_KEY, model=MODEL_NAME,
            max_model_len=CTX, keepalive_min=KEEPALIVE_MIN,
            tools="on" if with_tools else "text-only",
            startup_secs=int(time.time() - T0))

    t_serve = time.time()
    while time.time() - t_serve < KEEPALIVE_MIN * 60:
        time.sleep(120)
        if proc.poll() is not None:
            log(f"!! vLLM died (rc={proc.returncode}); last lines of {VLLM_LOG}:")
            tail = ""
            try:
                tail = "\n".join(VLLM_LOG.read_text(errors="replace").splitlines()[-25:])
                log("   " + tail.replace("\n", "\n   "))
            except Exception:  # noqa: BLE001
                pass
            publish("vllm-died", rc=proc.returncode, tail=tail[-800:])
            raise SystemExit(1)
        if tun_holder[0] is not None and tun_holder[0].poll() is not None:
            log("   tunnel process exited; opening a new one")
            new, tun = start_tunnel()
            if new:
                tun_holder[0] = tun
                publish("tunnel-url", endpoint=new, note="replaced")
                log(f"#  NEW ENDPOINT: {new}   (API key unchanged)")
        up_min = int((time.time() - t_serve) / 60)
        if up_min % 10 < 2:
            log(f"   heartbeat: up {up_min} min, endpoint {endpoint}")
            publish("heartbeat", up_min=up_min, endpoint=endpoint)
    log(f"   keepalive of {KEEPALIVE_MIN} min reached — shutting down")
    publish("auto-shutdown", served_min=KEEPALIVE_MIN)
    proc.terminate()


if __name__ == "__main__":
    main()
