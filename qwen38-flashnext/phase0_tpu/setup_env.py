#!/usr/bin/env python3
"""Phase 0b / TPU — step 1: environment (runs INSIDE the TPU session).

TPU-specific vs the CPU smoke test:
  - jax[tpu] 0.11.0 (libtpu comes with tpu-inference's own requirements path)
  - NO VLLM_TARGET_DEVICE=cpu and NO JAX_PLATFORMS=cpu — the TPU platform
    must activate natively
  - same ordering rule: torch is vllm's base, tpu-inference goes in --no-deps
"""

import os.path
import re
import subprocess
import sys

W = "/kaggle/working"
TPU_INF = f"{W}/tpu-inference"
FORK = f"{W}/nexus-tpu-fork"
PIN = "c824927"


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
    """True if the module imports (optionally at >= min_version) — a fast gate
    so re-runs don't pay pip's full resolution for already-installed stack."""
    try:
        m = __import__(mod)
        if min_version:
            from importlib.metadata import version
            v = version(mod)
            def vn(s):
                return [int(x) for x in re.findall(r"\d+", s)[:3]]
            return vn(v) >= vn(min_version)
        return True
    except Exception:
        return False


# 1. the torch base — but skip the heavy reinstall when a healthy 2.13.0 is
#    already in place (re-run friendliness; the TPU VM's preinstalled 2.8.0
#    gets force-replaced on the first run)
try:
    import torch as _torch
    _ok = _torch.__version__.startswith("2.13.0")
except Exception:
    _ok = False
if _ok:
    print("torch 2.13.0 already healthy — skipping force-reinstall")
else:
    sh('pip install -q --force-reinstall "torch==2.13.0" "torchvision==0.28.0"')
if not pip_have("vllm", "0.28.0"):
    sh('pip install -q "vllm==0.28.0"')
else:
    print("vllm 0.28.0 present — skipping")

if not os.path.isdir(TPU_INF):
    sh(f"git clone --quiet https://github.com/vllm-project/tpu-inference {TPU_INF}")
r = subprocess.run(f"git -C {TPU_INF} checkout --quiet --detach {PIN}", shell=True, capture_output=True)
if r.returncode != 0:
    sh(f"git -C {TPU_INF} fetch --quiet origin {PIN}")
    sh(f"git -C {TPU_INF} checkout --quiet --detach {PIN}")
sh(f"pip install -q -e {TPU_INF} --no-deps")

if not os.path.isdir(FORK):
    sh(f"git clone --quiet --depth 1 https://github.com/DQN-Labs/nexus-tpu-fork {FORK}")
sh(f"rm -rf {TPU_INF}/tpu_inference/models/jax/qwen4_exp")
sh(f"cp -r {FORK}/tpu_inference/models/jax/qwen4_exp {TPU_INF}/tpu_inference/models/jax/")
patch_target = f"{TPU_INF}/tpu_inference/models/common/model_loader.py"
r = subprocess.run(f"patch -p1 --dry-run -d {TPU_INF} < {FORK}/patches/model_loader.patch",
                   shell=True, capture_output=True)
if r.returncode == 0:
    sh(f"patch -p1 -d {TPU_INF} < {FORK}/patches/model_loader.patch")
elif os.path.isfile(patch_target) and "qwen4_exp" in open(patch_target).read():
    print("patch already applied — ok")
else:
    print("!! patch does not apply")
    raise SystemExit(1)

# the TPU jax stack: jax[tpu] brings libtpu; flax/tpu-info for the init chain
if pip_have("jax", "0.11.0") and pip_have("flax", "0.12.8"):
    print("jax/flax already healthy — skipping")
else:
    sh('pip install -q "jax[tpu]==0.11.0" "flax==0.12.8" "tpu-info==0.7.1" '
       "jaxtyping pytest pytest-mock absl-py safetensors numpy")

# torchax must match OUR torch (2.13.0): the VM's preinstalled copy targets the
# old image torch (2.8) and dies at `torch.ops.aten.prod.dim_Dimname` on
# import (seen live). The Sept-18 dev wheel is built against torch 2.13;
# --no-deps keeps it from touching torch/jax.
sh('pip install -q --no-deps --upgrade "torchax==0.0.14.dev20260918"')

sh("pip uninstall -q -y tpu-inference-qwen4exp || true")

import importlib  # noqa: E402
import traceback  # noqa: E402
sys.path.insert(0, TPU_INF)
importlib.invalidate_caches()
try:
    import torch  # noqa: E402,F401
    import vllm  # noqa: E402,F401
    import jax  # noqa: E402,F401
    import tpu_inference  # noqa: E402
    from tpu_inference.models.jax.qwen4_exp import weight_loader as WL  # noqa: E402,F401
except Exception:
    print("!! import failed — full chain below:")
    traceback.print_exc()
    raise SystemExit(1)
print("torch:", torch.__version__, "| vllm:", vllm.__version__, "| jax:", jax.__version__)
print("jax devices:", jax.devices())
assert len(jax.devices()) == 8, "expected 8 TPU devices"
print("STEP 1 OK — environment ready")
