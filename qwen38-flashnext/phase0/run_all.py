#!/usr/bin/env python3
"""Phase 0 driver — run every smoke-test cell in order and tee output to a log.

Run from the one-cell notebook runner:
    python /kaggle/working/kaggle-tpu-lab/qwen38-flashnext/phase0/run_all.py

Each cell is executed as its own script (single source of truth with the
copy-paste flow); its output is forwarded line-by-line through a tee so the
console AND /kaggle/working/phase0_results.txt both capture it. Cell 2
(install) gates cells 3-4; cell 5 (shard probe) is independent and always runs.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/kaggle/working/phase0_results.txt")


class Tee:
    """stdout + log file, so the user can copy one file back to me."""

    def __init__(self, path):
        self.file = open(path, "w", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def run_logged(script):
    """Run a cell script; forward its output line-by-line through stdout (the Tee)."""
    p = subprocess.Popen([sys.executable, str(script)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    return p.returncode


sys.stdout = Tee(LOG)
started = time.time()

CELLS = [
    ("cell_01_env.py", "MACHINE REPORT", "info"),
    ("cell_02_install.py", "INSTALL (tpu-inference pinned + fork overlay)", "gates"),
    ("cell_03_tests.py", "FORK CPU TESTS (expect 15 passed)", "needs-install"),
    ("cell_04_loader_smoke.py", "WEIGHT-LOADER AUDIT vs the real checkpoint", "needs-install"),
    ("cell_05_shard_probe.py", "SHARD HEADERS -> HBM vs host memory split", "independent"),
]

results = {}
for fname, title, kind in CELLS:
    if kind == "needs-install" and results.get("cell_02_install.py") != 0:
        print(f"\n### skipping {fname}: install failed, nothing to run it against")
        results[fname] = "skipped"
        continue
    print("\n" + "#" * 70)
    print(f"### {title}")
    print("#" * 70)
    results[fname] = run_logged(HERE / fname)
    print(f"### {fname} -> rc={results[fname]}")

print("\n" + "=" * 70)
print(f"PHASE 0 SUMMARY  ({(time.time() - started) / 60:.1f} min total)")
for fname, title, _ in CELLS:
    rc = results.get(fname)
    mark = {0: "OK", "skipped": "SKIPPED"}.get(rc, "FAILED" if rc else "OK")
    print(f"  {mark:<8} {fname:<28} {title}")
print("=" * 70)
print(f"Everything above is also saved to: {LOG}")
print("Send me the whole output (or that file).")
sys.stdout.file.close()
sys.stdout = sys.__stdout__
