#!/usr/bin/env python3
"""qwen38-27b-gpu / phase0 — driver: environment + checkpoint. HALTS before
the load; load_and_bench runs as its own step so a first-load failure never
costs the environment work."""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/kaggle/working/phase0gpu_results.txt")

# The driver's children inherit this env, and the fix has to be in place
# before setup_env.py imports vllm.
sys.path.insert(0, str(HERE))
import cu_env  # noqa: E402

cu_env.reexec()


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def run_logged(script):
    p = subprocess.Popen([sys.executable, str(script)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    return p.returncode


sys.stdout = Tee(LOG)
started = time.time()

for name, title in [
    ("setup_env.py", "GPU ENVIRONMENT (torch+vllm in the driver-matched CUDA build)"),
    ("resolve_checkpoint.py", "CHECKPOINT RESOLVER (datasets else hf_transfer)"),
]:
    print("\n" + "#" * 70)
    print(f"### {title}")
    print("#" * 70)
    rc = run_logged(HERE / name)
    print(f"### {name} -> rc={rc}")
    if rc != 0:
        print(f"PHASE 0B-GPU HALTED at {name} — send the output above")
        sys.stdout.file.close()
        sys.stdout = sys.__stdout__
        raise SystemExit(1)

print("\n" + "=" * 70)
print(f"PHASE 0B-GPU READY  ({(time.time() - started) / 60:.1f} min)")
print("checkpoint view: /kaggle/tmp/ckpt")
print("=" * 70)
print("Load + proof + benchmark: run load_and_bench.py (cell 4).")
sys.stdout.file.close()
sys.stdout = sys.__stdout__
