# CELL 3 — verify the pinned llama-server build supports every flag the endpoint
# uses, BEFORE deploying. CPU-only, costs cents. Pinned to b11379 (latest release,
# 2026-10-01; includes qwen4exp + GSQ quant support).
# If the image pull fails with "manifest unknown", swap IMAGE to
# "ghcr.io/ggml-org/llama.cpp:server-cuda" (unpinned latest) and re-run.

import pathlib
import subprocess

PROBE = '''
import modal

IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda-b11379"

image = modal.Image.from_registry(IMAGE, add_python="3.12").entrypoint([])
app = modal.App("flashnext-probe", image=image)

WANT = [
    "--spec-type", "--model-draft", "--spec-draft-n-max",  # MTP speculative decoding
    "--lazy-mode", "-lm",                                  # mmap n-gram shard
    "--api-key", "--jinja", "--alias",                     # serving
    "--n-gpu-layers", "--ctx-size", "--flash-attn",        # GPU + context
    "--cache-type-k", "--cache-type-v", "--mmproj",        # KV quant + vision
    "--parallel", "--host", "--port",
]


@app.function(cpu=1, timeout=300)
def flags():
    import subprocess

    v = subprocess.run(["/app/llama-server", "--version"], capture_output=True, text=True)
    print("version:", (v.stdout.strip() or v.stderr.strip())[:200])
    out = subprocess.run(["/app/llama-server", "--help"], capture_output=True, text=True).stdout
    lines = out.splitlines()
    misses = []
    for w in WANT:
        hit = [l.strip()[:100] for l in lines if w in l]
        print(("OK  " if hit else "MISS"), w, "|", hit[0] if hit else "")
        if not hit:
            misses.append(w)
    print("PROBE_DONE misses=" + (",".join(misses) if misses else "none"))


@app.local_entrypoint()
def main():
    flags.remote()
'''

pathlib.Path("probe_server.py").write_text(PROBE)
compile(PROBE, "probe_server.py", "exec")
print("wrote probe_server.py - running")
subprocess.run(["modal", "run", "probe_server.py"], check=True)
