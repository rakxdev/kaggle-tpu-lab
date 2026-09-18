# Phase 0 / cell 3 — run the fork's 15 CPU unit tests
# These construct tiny random models and run every component (config, name
# mapping, GDN, QSA, PLE, MoE, Q4 dequant) on CPU devices. No TPU needed —
# that is the point: engine soundness before any quota is spent.
# Expect the summary line:  "15 passed"

import subprocess

# tests must import the OVERLAYED checkout (with qwen4_exp + patch), so pin it
# on PYTHONPATH; JAX_PLATFORMS=cpu keeps libtpu from hijacking jax on CPU
r = subprocess.run(
    "cd /kaggle/working/nexus-tpu-fork && "
    "JAX_PLATFORMS=cpu VLLM_TARGET_DEVICE=cpu PYTHONPATH=/kaggle/working/tpu-inference "
    "python -m pytest tests/models/jax/test_qwen4_exp.py -q",
    shell=True, text=True, capture_output=True, timeout=3600,
)
out = (r.stdout + r.stderr).strip().splitlines()
for line in out[-30:]:
    print(line)

last = out[-1] if out else ""
if "passed" in last and "failed" not in last and "error" not in last:
    print("CELL 3 OK —", last)
else:
    print("CELL 3 INVESTIGATE — summary above; send me the last 30 lines")
