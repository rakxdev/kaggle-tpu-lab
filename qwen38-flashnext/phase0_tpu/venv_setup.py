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
    if not (sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip-uv")
            and sh([sys.executable, "-m", "uv", "venv", VENV,
                    "--python", sys.executable], "uv-venv")):
        print("!! uv venv failed")
        raise SystemExit(1)
    # One resolve for the whole stack: uv picks a consistent set (CPU torch —
    # a TPU never uses the CUDA wheels, same as vllm-tpu's own Docker image).
    pkgs = ["--torch-backend=cpu", "vllm==0.28.0", "tpu-inference==0.28.0",
            "jax[tpu]==0.11.0"]
    if not sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
               *pkgs], "uv-install"):
        print("!! venv package install failed")
        raise SystemExit(1)

    print("=" * 68)
    print("STEP 2  fork overlay inside the venv (fork_apply.py verbatim)")
    print("=" * 68)
    r = subprocess.run([PY, "-c",
                        "import tpu_inference, os; "
                        "print(os.path.dirname(tpu_inference.__file__))"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("!! cannot locate tpu_inference in the venv:", r.stderr[-300:])
        raise SystemExit(1)
    base = Path(r.stdout.strip())
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
