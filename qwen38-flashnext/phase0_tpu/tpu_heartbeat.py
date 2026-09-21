#!/usr/bin/env python3
"""TPU heartbeat — never lose a queue slot to Kaggle's idle-accelerator stop.

Kaggle stops accelerator sessions that show no utilisation, and everything we
run BEFORE the first real load (env build, installs, failure triage) is
CPU-side. Seen live 2026-09-20: a v5e-8 session died to the idle detector
while the load was being unblocked, costing a ~7-hour queue position.

So while the bench is NOT holding the TPU, touch it every few minutes from a
short-lived subprocess (system python has a working jax on the TPU image).
The touch releases the chips on exit. Two guards prevent the heartbeat from
stealing /dev/vfio/0 from the engine (the /dev/vfio lesson):
  1. load_and_bench.py writes /tmp/tpu_in_use before touching jax;
  2. a load_and_bench process being alive also mutes the touch.
If both say the coast is clear but the engine grabs the device first anyway,
the touch fails with "Device or resource busy", is logged, and the loop
continues — harmless.
"""

import os
import subprocess
import sys
import time

FLAG = "/tmp/tpu_in_use"
INTERVAL = int(os.environ.get("TPU_HEARTBEAT_S", "240"))
# system python: the TPU image ships a working jax (proven in every probe)
PY = sys.executable
TOUCH = ("import jax; x = jax.numpy.ones((8, 8)); "
         "print('HB', float(x.sum()))")


def bench_alive():
    r = subprocess.run(["pgrep", "-f", "load_and_bench"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


while True:
    if os.path.exists(FLAG) or bench_alive():
        print(time.strftime("[%H:%M:%S]"), "heartbeat: bench owns the TPU — skipping",
              flush=True)
    else:
        try:
            r = subprocess.run([PY, "-c", TOUCH], capture_output=True,
                               text=True, timeout=180)
            if "HB" in (r.stdout or ""):
                print(time.strftime("[%H:%M:%S]"), "heartbeat: TPU touched", flush=True)
            else:
                tail = ((r.stderr or "").strip().splitlines() or ["?"])[-1]
                print(time.strftime("[%H:%M:%S]"), f"heartbeat: touch failed — {tail[-120:]}",
                      flush=True)
        except Exception as e:  # noqa: BLE001
            print(time.strftime("[%H:%M:%S]"), f"heartbeat: error {e!r}", flush=True)
    time.sleep(INTERVAL)
