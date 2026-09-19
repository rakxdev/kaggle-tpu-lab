#!/usr/bin/env python3
"""qwen38-27b-gpu / phase0 — step 1: TPU-free, jax-free environment.

Hardware target: Kaggle 2x T4 (SM75, 16 GB each). Stack: vllm 0.28.0
(Qwen3.8-27B stock support) + torch 2.13.0 in the CUDA variant that matches
the VM's driver (auto-detected from nvidia-smi; the 67ailab T4 benchmark
needed cu121 wheels on older drivers — we do not guess).

Rules encoded from today's failures:
  - no jax anywhere in this route (nothing imports it)
  - heavy work runs in THIS script's process, not notebook `!` magic
  - every pip step is gated by an import pre-check (re-runs cost seconds)
  - the checkpoint resolver is hybrid: mounted datasets else parallel HF
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cu_env  # noqa: E402

# Kaggle's T4 image is a CUDA-12.8 host; vllm 0.28.0 is a cu130 build whose
# runtime the loader cannot see. Re-exec fixes the path before `import vllm`.
cu_env.reexec()

VLLM_PIN = "0.28.0"
TORCH_PIN = "2.13.0"
INT4_REPO = "RedHatAI/Qwen3.8-27B-INT4"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    for line in (r.stdout + r.stderr).strip().splitlines()[-6:]:
        print("   ", line)
    if r.returncode != 0:
        print(f"!! FAILED (rc={r.returncode})")
        raise SystemExit(1)
    return r


def pip_have(mod, min_version=None):
    try:
        __import__(mod)
        if min_version:
            from importlib.metadata import version

            def vn(s):
                return [int(x) for x in re.findall(r"\d+", s)[:3]]
            return vn(version(mod)) >= vn(min_version)
        return True
    except Exception:
        return False


def driver_cuda():
    """Max CUDA version the VM's driver supports (nvidia-smi's 'CUDA Version')."""
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return None, out[:400]
    return (int(m.group(1)), int(m.group(2))), out


def torch_cuda_variant(driver_cuda):
    """Pick the torch/vllm CUDA build the driver can run. CUDA 12.x has
    minor-version compatibility (any 12.x runtime runs on >=525 drivers);
    CUDA 13 needs a 580+ driver."""
    if driver_cuda is None:
        return "cu130"  # default PyPI build; probe will catch problems
    major, minor = driver_cuda
    if major >= 13:
        return "cu130"
    if major == 12:
        if minor >= 9:
            return "cu129"
        if minor >= 8:
            return "cu128"
        if minor >= 6:
            return "cu126"
        return "cu121"
    return "cu121"


def main():
    # 0. GPUs must exist (rows say "Tesla T4"; the header's single "NVIDIA-SMI"
    #    is why counting "NVIDIA" saw 1 instead of 2 — seen live)
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    gpus = len(re.findall(r"Tesla T4", out))
    print(out[:1200], flush=True)
    if gpus < 2:
        print(f"!! expected 2 T4s, nvidia-smi shows {gpus}")
        raise SystemExit(1)

    drv, _ = driver_cuda()
    variant = torch_cuda_variant(drv)
    print(f"driver CUDA: {drv} -> wheel variant: {variant}", flush=True)

    # 1. torch in the matching CUDA build (skip when already correct)
    try:
        import torch as _t
        _ok = _t.__version__.startswith(TORCH_PIN) and _t.cuda.is_available() \
            and _t.cuda.device_count() == 2
    except Exception:
        _ok = False
    if _ok:
        print(f"torch {TORCH_PIN} already healthy on 2 GPUs — skipping")
    else:
        if variant in ("cu126", "cu128", "cu129", "cu121"):
            sh(f'pip install -q --force-reinstall "torch=={TORCH_PIN}" '
               f'"torchvision==0.28.0" --index-url '
               f"https://download.pytorch.org/whl/{variant}")
        else:
            sh(f'pip install -q --force-reinstall "torch=={TORCH_PIN}" '
               '"torchvision==0.28.0"')

    # 2. vllm (the arch is supported stock; marlin W4A16 works on SM75)
    if not pip_have("vllm", VLLM_PIN):
        sh(f'pip install -q "vllm=={VLLM_PIN}"')
    else:
        print("vllm already present — skipping")

    # 3. transformers >= 5.8.0 (the recipe's requirement for the Qwen3.5
    #    processor classes) + the fast downloader + tokenizer stack
    sh('pip install -q -U "transformers>=5.8.0" "huggingface_hub" hf_transfer '
       "safetensors")

    # 4. import probe: torch sees 2 GPUs, CUDA initializes for real, vllm imports
    import torch  # noqa: E402
    import vllm  # noqa: E402,F401
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2, \
        f"CUDA not usable: {torch.cuda.device_count()} devices"
    x = torch.randn(64, 64, device="cuda:1") @ torch.randn(64, 64, device="cuda:1")
    torch.cuda.synchronize()
    print("torch:", torch.__version__, "| vllm:", vllm.__version__,
          "| gpus:", torch.cuda.device_count(), "| matmul ok:", float(x.sum()) != 0)

    print("STEP 1 OK — environment ready", flush=True)


if __name__ == "__main__":
    main()
