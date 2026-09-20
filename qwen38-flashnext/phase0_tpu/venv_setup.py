#!/usr/bin/env python3
"""Flash-Next Phase 0b — runtime via the PROVEN isolated-venv pattern.

The system-python pip surgery (torch upgrades, --no-deps dances, constraint
pins) hit three version skews in a row on 2026-09-20. Our working TPU kernels
(serve_qwen38.py / serve_glm53.py) never touch the system stack: they build a
uv venv at /tmp/venv with pinned packages (~30 s warm) and run everything with
the venv's python. This is that recipe with the fork's package set, resolved
TOGETHER so pip cannot produce a skew:

    vllm==0.28.0 + tpu-inference==0.28.0 + jax[tpu]==0.11.0

then the fork overlay (qwen4_exp leaf + startup .pth) inside the VENV's
site-packages — fork_apply.py's mechanism, verbatim. The notebook kernel never
imports jax; every check runs in a subprocess that releases the TPU on exit.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
FORK = "/kaggle/working/nexus-tpu-fork"

# The venv ships its own libtpu; don't let the image's TPU_LIBRARY_PATH
# override it (serve_qwen38.py, line ~101).
os.environ.pop("TPU_LIBRARY_PATH", None)


# vllm's runtime deps, curated by the fork itself (scripts/kaggle_cells_bench/
# setup.py, their proven serving set). vllm==0.28.0 goes in --no-deps because
# its full tree pins a numba that conflicts with tpu-inference's (seen live:
# "tpu-inference 0.28.0 and vllm==0.28.0 are incompatible ... numba==0.65.0");
# this list is what the fork served with. Deliberately NOT installed (their
# note): CUDA-only kernels (flashinfer/tilelang/cutlass/quack/tokenspeed/
# humming/torchcodec/PyNvVideoCodec/tvm-ffi) — the JAX/TPU path never runs
# them; add on demand if a traceback names one.
VLLM_RUNTIME_DEPS = [
    # server + sampling orchestration
    "fastapi", "uvicorn", "openai", "pydantic", "tiktoken", "sentencepiece",
    "safetensors", "tokenizers", "einops", "cloudpickle", "msgspec", "pyzmq",
    "setproctitle", "psutil", "pillow", "protobuf", "python-json-logger",
    "prometheus_client", "blake3", "py-cpuinfo", "partial-json-parser",
    "jsonschema", "filelock", "pyyaml",
    # vllm hard imports (the fork diagnosed these from its server log)
    "openai-harmony", "anthropic", "model-hosting-container-standards", "mcp",
    "opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp",
    "opentelemetry-semantic-conventions-ai", "ninja", "cachetools", "cbor2",
    "ijson", "pybase64", "compressed-tensors==0.17.0", "fastsafetensors",
    "outlines_core==0.2.14", "lm-format-enforcer==0.11.3", "xgrammar",
    "llguidance", "mistral_common", "depyf",
    "prometheus-fastapi-instrumentator",
    "lark==1.2.2", "huggingface_hub>=1.27.0",
    # fresh venv: transformers normally preinstalled on the image must be ours
    "transformers>=5.8.0",
]


def sh(cmd, tag, timeout=2400):
    t0 = time.time()
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-4:]:
        print("   ", ln, flush=True)
    print(f"   [{tag}] rc={r.returncode} in {time.time() - t0:.0f}s", flush=True)
    return r.returncode == 0


VERIFY = r"""
import importlib.metadata as md
for p in ("tpu-inference", "torchax", "jax", "vllm", "torch", "transformers"):
    try: print("VER", p, md.version(p))
    except Exception: print("VER", p, "ABSENT")
from tpu_inference.models.jax.qwen4_exp.startup import install
print("INSTALL_OK", sorted(install()))
import vllm
import vllm.entrypoints.cli.main  # the fork's own smoke test: full CLI chain
print("VLLM_IMPORT_OK", vllm.__version__)
import jax
n = len(jax.devices())
print("DEVICES", n)
assert n == 8, "expected 8 TPU devices"
print("VENV_VERIFY_OK")
"""


def main():
    t0 = time.time()
    print("=" * 68)
    print("STEP 1  uv venv at", VENV, "(the serve_qwen38.py recipe)")
    print("=" * 68)
    shutil.rmtree(VENV, ignore_errors=True)  # rebuild fresh, every time
    if not (sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip-uv")
            and sh([sys.executable, "-m", "uv", "venv", VENV,
                    "--python", sys.executable], "uv-venv")):
        print("!! uv venv failed")
        raise SystemExit(1)
    # 1a. the engine stack: tpu-inference with its deps (torchax 0.0.13 +
    #     numba + CPU torch via --torch-backend), jax[tpu] pinned. One resolve.
    if not sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
               "--torch-backend=cpu", "tpu-inference==0.28.0",
               "jax[tpu]==0.11.0"], "uv-engine"):
        print("!! engine stack install failed")
        raise SystemExit(1)
    # 1b. torchaudio matched to whatever torch landed — AND from the CPU
    #     index: PyPI's default torchaudio wheel is CUDA-built and its native
    #     lib fails to dlopen against CPU torch (seen live: "Could not load
    #     this library: .../torchaudio/lib/libtorchaudio.so").
    r = subprocess.run([PY, "-c",
                        "import importlib.metadata as md; "
                        "print('.'.join(md.version('torch').split('.')[:2]))"],
                       capture_output=True, text=True)
    tmaj = r.stdout.strip()
    print(f"   torch in venv: {tmaj}.x — pinning torchaudio (CPU build) to match", flush=True)
    if not sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
               "--index-url", "https://download.pytorch.org/whl/cpu",
               f"torchaudio=={tmaj}.0+cpu"], "uv-torchaudio"):
        print("!! torchaudio pin failed (non-fatal — continuing)")
    # 1c. vllm --no-deps (its full tree conflicts with tpu-inference on numba)
    #     + the fork's curated runtime set.
    if not (sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
                "--no-deps", "vllm==0.28.0"], "uv-vllm")
            and sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
                    *VLLM_RUNTIME_DEPS], "uv-vllm-deps")):
        print("!! vllm install failed")
        raise SystemExit(1)

    print("=" * 68)
    print("STEP 2  fork overlay inside the venv (fork_apply.py verbatim)")
    print("=" * 68)
    r = subprocess.run([PY, "-c",
                        "import sysconfig; "
                        "print(sysconfig.get_paths()['purelib'])"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("!! cannot locate the venv's site-packages:", r.stderr[-300:])
        raise SystemExit(1)
    base = Path(r.stdout.strip()) / "tpu_inference"
    src = Path(FORK) / "tpu_inference" / "models" / "jax" / "qwen4_exp"
    dst = base / "models" / "jax" / "qwen4_exp"
    if not src.is_dir():
        print(f"!! fork leaf missing at {src} — clone the fork first")
        raise SystemExit(1)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    print(f"   overlay {src} -> {dst}", flush=True)
    pth = base.parent / "qwen4exp_tpu_startup.pth"
    pth.write_text("import tpu_inference.models.jax.qwen4_exp.startup\n")
    print(f"   startup hook -> {pth}", flush=True)

    print("=" * 68)
    print("STEP 3  verify with the venv python (subprocess — releases the TPU)")
    print("=" * 68)
    r = subprocess.run([PY, "-c", VERIFY], text=True, capture_output=True,
                       timeout=900)
    for ln in (r.stdout or "").splitlines():
        print("   ", ln, flush=True)
    if r.returncode != 0 or "VENV_VERIFY_OK" not in (r.stdout or ""):
        print("!! venv verify failed:")
        print((r.stderr or "")[-1200:])
        raise SystemExit(1)

    print("=" * 68)
    print(f"VENV READY ({(time.time() - t0) / 60:.1f} min) — run everything with:")
    print(f"  {PY} qwen38-flashnext/phase0_tpu/load_and_bench.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
