#!/usr/bin/env python3
"""Phase 0b driver — environment + checkpoint view. STOPS before loading:
the load/benchmark is driven interactively over the relay."""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/kaggle/working/phase0b_results.txt")


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

STEPS = [
    ("machine_report", None),
    ("setup_env.py", "TPU ENVIRONMENT (installs ~15 min)"),
    ("mount_weights.py", "CHECKPOINT VIEW (13 datasets -> one dir)"),
]

for name, title in STEPS:
    print("\n" + "#" * 70)
    if title is None:  # machine report
        subprocess.run("nproc; free -g | head -2; df -h /kaggle/working /kaggle/tmp / 2>/dev/null | head -6", shell=True)
        subprocess.run("ls /kaggle/input | grep qwen38-flashnext | head -20", shell=True)
        continue
    print(f"### {title}")
    print("#" * 70)
    rc = run_logged(HERE / name)
    print(f"### {name} -> rc={rc}")
    if rc != 0:
        print(f"PHASE 0B HALTED at {name} — send the output above")
        sys.stdout.file.close()
        sys.stdout = sys.__stdout__
        raise SystemExit(1)

import jax  # noqa: E402
print("\n" + "=" * 70)
print(f"PHASE 0B READY  ({(time.time() - started) / 60:.1f} min)")
print("devices:", jax.devices())
print("checkpoint view: /kaggle/tmp/ckpt")
print("=" * 70)
print("Load + proof + benchmark are driven via the relay from here.")
sys.stdout.file.close()
sys.stdout = sys.__stdout__
