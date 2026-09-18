# Phase 0 / cell 2 — install the stack (CPU session, ~10–20 min first run)
# 1. tpu-inference at the commit the fork was written against (c824927)
# 2. the fork's qwen4_exp JAX model copied on top + its registration patch
# 3. import check against the OVERLAYED tree (never a fork editable install —
#    the fork's pyproject maps `tpu_inference` onto its partial tree and
#    shadows the real checkout, seen live).
# Every step is idempotent: safe to re-run this cell any number of times.

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


# 1. jax CPU first (the version the fork targets; libtpu is NOT wanted here)
sh("pip install -q 'jax[cpu]==0.11.0' flax pytest safetensors numpy")

# 2. tpu-inference at the pinned commit (full clone contains every ancestor,
#    so a plain detach checkout works; fetch-by-sha is what GitHub rejects)
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
sh(f"pip install -q -e {TPU_INF}")

# 2b. vllm is a PEER dependency, not in tpu-inference's requirements.txt —
# without it the package __init__ dies at `from vllm.logger import ...`
# (seen live). Install it separately: co-resolving both in one pip call is
# impossible at this pin (tpu-inference pins numba==0.62.1, vllm 0.28.0 wants
# 0.65.0); installed sequentially, pip upgrades numba and only warns about
# tpu-inference's stale pin — expected and harmless for these CPU tests.
sh("pip install -q 'vllm==0.28.0'")

# 3. fork overlay: refresh the copy (rm first — a plain re-cp would nest), then
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

# 3b. repair: an earlier `pip install -e fork` (if run) shadows tpu_inference
sh("pip uninstall -q -y tpu-inference-qwen4exp || true")

# 4. import check — resolve explicitly to the overlayed checkout so the
#    running interpreter does not depend on pip's freshly-written .pth files.
#    VLLM_TARGET_DEVICE=cpu is authoritative in vllm's platform resolver: it
#    skips plugin probing entirely — otherwise tpu-inference's TPU platform
#    plugin activates on detection paths that cannot work on a CPU box and
#    the lazy `current_platform` resolution dies inside __getattr__ (seen
#    live as "cannot import name 'current_platform'").
import importlib
import traceback
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
sys.path.insert(0, TPU_INF)
importlib.invalidate_caches()
try:
    import vllm  # noqa: E402,F401
    import tpu_inference  # noqa: E402
    from tpu_inference.models.jax.qwen4_exp import weight_loader as WL  # noqa: E402
except Exception:
    print("!! import failed — full underlying chain below (send it back):")
    traceback.print_exc()
    raise SystemExit(1)
assert WL.__file__ and TPU_INF in WL.__file__, f"wrong tree: {WL.__file__}"
print("vllm:", vllm.__version__, "| platform:", os.environ["VLLM_TARGET_DEVICE"])
print("tpu-inference:", tpu_inference.__file__)
print("qwen4_exp weight_loader:", WL.__file__)

print("CELL 2 OK")
