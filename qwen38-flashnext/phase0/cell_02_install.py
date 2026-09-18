# Phase 0 / cell 2 — install the stack (CPU session, ~10–20 min first run)
# 1. tpu-inference at the commit the fork was written against (c824927)
# 2. the fork's qwen4_exp JAX model copied on top + its registration patch
# 3. fork + test deps
# Everything lands in /kaggle/working (20 GB limit; the checkout is ~1 GB).

import os
import subprocess

W = "/kaggle/working"
TPU_INF = f"{W}/tpu-inference"
FORK = f"{W}/nexus-tpu-fork"
PIN = "c824927"  # tpu-inference commit the fork's docs target


def sh(cmd):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    for line in tail[-8:]:
        print("   ", line)
    if r.returncode != 0:
        print(f"!! FAILED (rc={r.returncode}) — full output kept in the cell log")
        raise SystemExit(1)
    return r


# 1. jax CPU first (the version the fork targets; libtpu is NOT wanted here)
sh("pip install -q 'jax[cpu]==0.11.0' flax pytest safetensors numpy")

# 2. tpu-inference at the pinned commit
if not os.path.isdir(TPU_INF):
    sh(f"git clone --quiet https://github.com/vllm-project/tpu-inference {TPU_INF}")
# shallow clone cannot always check out a raw SHA: try cheap fetch, fall back to full history
r = subprocess.run(f"git -C {TPU_INF} fetch --depth 1 origin {PIN}", shell=True, capture_output=True)
if r.returncode == 0:
    sh(f"git -C {TPU_INF} checkout --quiet --detach {PIN}")
else:
    print("shallow fetch of the pin failed — unshallowing (one-time, slower)")
    sh(f"git -C {TPU_INF} fetch --quiet --unshallow")
    sh(f"git -C {TPU_INF} checkout --quiet --detach {PIN}")
print("tpu-inference at:", subprocess.run(f"git -C {TPU_INF} rev-parse --short HEAD",
                                          shell=True, capture_output=True, text=True).stdout.strip())
sh(f"pip install -q -e {TPU_INF}")

# 3. fork overlay: copy the model in, apply the registration patch (loudly)
if not os.path.isdir(FORK):
    sh(f"git clone --quiet --depth 1 https://github.com/DQN-Labs/nexus-tpu-fork {FORK}")
sh(f"cp -r {FORK}/tpu_inference/models/jax/qwen4_exp {TPU_INF}/tpu_inference/models/jax/")
r = subprocess.run(f"patch -p1 --dry-run -d {TPU_INF} < {FORK}/patches/model_loader.patch",
                   shell=True, capture_output=True)
if r.returncode != 0:
    print("!! the fork's patch does not apply to this tpu-inference checkout:")
    print(r.stdout.decode()[-500:], r.stderr.decode()[-500:])
    raise SystemExit(1)
sh(f"patch -p1 -d {TPU_INF} < {FORK}/patches/model_loader.patch")
sh(f"pip install -q -e '{FORK}[test]'")

# 4. import check (the overlay registers into tpu-inference's registry)
import tpu_inference  # noqa: E402
from tpu_inference.models.jax.qwen4_exp import weight_loader as WL  # noqa: E402
print("tpu-inference:", tpu_inference.__file__)
print("qwen4_exp weight_loader:", WL.__file__)

print("CELL 2 OK")
