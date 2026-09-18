# Phase 0 / cell 2 — install the stack (CPU session, ~10–20 min first run)
#
# Ordering rule (learned the hard way, seen live):
#   vllm 0.28.0 and tpu-inference@pin have irreconcilable pins on torch /
#   torchvision / numba. Installing them naively in either order makes pip
#   swap torch underneath the other and leaves a hybrid native tree
#   ("libtorch_cuda.so: undefined symbol: ncclCommResume").
#   So: vllm's torch (2.13.0) is THE base — installed first and force-repaired;
#   tpu-inference goes in with --no-deps so it can never touch torch again;
#   the jax-side packages its init chain needs are installed explicitly.
# Every step is idempotent: safe to re-run any number of times.

import os
import subprocess
import sys

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
        print(f"!! FAILED (rc={r.returncode})")
        raise SystemExit(1)
    return r


# 1. the torch base: force-reinstall repairs a hybrid tree from any earlier
#    swap (fresh .so files + matching nvidia-nccl) and is a plain install on
#    a fresh session
sh('pip install -q --force-reinstall "torch==2.13.0" "torchvision==0.28.0"')

# 2. vllm on top of it (its other deps: transformers, fastapi, numba 0.65 ...)
sh('pip install -q "vllm==0.28.0"')

# 3. tpu-inference at the pinned commit — code + entry points only (--no-deps!)
if not os.path.isdir(TPU_INF):
    sh(f"git clone --quiet https://github.com/vllm-project/tpu-inference {TPU_INF}")
r = subprocess.run(f"git -C {TPU_INF} checkout --quiet --detach {PIN}", shell=True, capture_output=True)
if r.returncode != 0:
    print(f"direct checkout of {PIN} failed — trying an explicit fetch")
    r2 = subprocess.run(f"git -C {TPU_INF} fetch --quiet origin {PIN}", shell=True, capture_output=True)
    if r2.returncode != 0:
        print("!! could not check out the pin; continuing on HEAD — the patch "
              "dry-run below is the real compatibility gate")
        print(r2.stderr.decode()[-300:])
    else:
        sh(f"git -C {TPU_INF} checkout --quiet --detach {PIN}")
head = subprocess.run(f"git -C {TPU_INF} rev-parse --short HEAD",
                      shell=True, capture_output=True, text=True).stdout.strip()
print("tpu-inference checked out at:", head, "(pin:", PIN + ")")
sh(f"pip install -q -e {TPU_INF} --no-deps")

# 4. the jax-side packages the tpu_inference init chain + the fork need
#    (tpu-inference's own requirements would downgrade torch — never install
#    them wholesale)
sh('pip install -q "jax[cpu]==0.11.0" "flax==0.12.8" "tpu-info==0.7.1" '
   "jaxtyping pytest pytest-mock absl-py safetensors numpy")

# 5. fork overlay: refresh the copy (rm first — a plain re-cp would nest), then
#    apply the registration patch, tolerating an already-patched tree
if not os.path.isdir(FORK):
    sh(f"git clone --quiet --depth 1 https://github.com/DQN-Labs/nexus-tpu-fork {FORK}")
sh(f"rm -rf {TPU_INF}/tpu_inference/models/jax/qwen4_exp")
sh(f"cp -r {FORK}/tpu_inference/models/jax/qwen4_exp {TPU_INF}/tpu_inference/models/jax/")
patch_target = f"{TPU_INF}/tpu_inference/models/common/model_loader.py"
r = subprocess.run(f"patch -p1 --dry-run -d {TPU_INF} < {FORK}/patches/model_loader.patch",
                   shell=True, capture_output=True)
if r.returncode == 0:
    sh(f"patch -p1 -d {TPU_INF} < {FORK}/patches/model_loader.patch")
elif os.system(f"grep -q qwen4_exp {patch_target}") == 0:
    print("patch already applied — ok")
else:
    print("!! the fork's patch does not apply to this tpu-inference checkout:")
    print(r.stdout.decode()[-400:], r.stderr.decode()[-400:])
    raise SystemExit(1)

# 5b. repair: an earlier `pip install -e fork` (if run) shadows tpu_inference
sh("pip uninstall -q -y tpu-inference-qwen4exp || true")

# 6. import check — resolve explicitly to the overlayed checkout so the
#    running interpreter does not depend on pip's freshly-written .pth files.
#    VLLM_TARGET_DEVICE=cpu is authoritative in vllm's platform resolver: it
#    skips plugin probing — otherwise tpu-inference's TPU platform plugin
#    activates on detection paths that cannot work on a CPU box and the lazy
#    `current_platform` resolution dies inside __getattr__ (seen live).
import importlib
import traceback
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
sys.path.insert(0, TPU_INF)
importlib.invalidate_caches()
try:
    import torch  # noqa: E402,F401
    import vllm  # noqa: E402,F401
    import jax  # noqa: E402,F401
    import tpu_inference  # noqa: E402
    from tpu_inference.models.jax.qwen4_exp import weight_loader as WL  # noqa: E402
except Exception:
    print("!! import failed — full underlying chain below (send it back):")
    traceback.print_exc()
    raise SystemExit(1)
assert WL.__file__ and TPU_INF in WL.__file__, f"wrong tree: {WL.__file__}"
print("torch:", torch.__version__, "| vllm:", vllm.__version__,
      "| jax:", jax.__version__, "| jax devices:", jax.devices())
print("tpu-inference:", tpu_inference.__file__)
print("qwen4_exp weight_loader:", WL.__file__)

print("CELL 2 OK")
