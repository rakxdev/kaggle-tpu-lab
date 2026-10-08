#!/usr/bin/env python3
# CELL 3 — download the two GGUFs (background; poll with cell_03b_status.sh).
#
# Cost: ~10-20 min of quota at Kaggle→HF speeds. 13.2 GB total:
#   Escha-Qwen3.8-27B-W2-Q8E.gguf           10,307,703,008 bytes  (the main model)
#   Escha-Qwen3.8-27B-W2-MTP-F16-headQ4.gguf 2,926,418,048 bytes  (the MTP draft)
# The 12.69 GB F16-head build is deliberately skipped — the author measured
# Q8_0 as both smaller AND faster, and 99.4% top-1 agreement with his fp32
# reference build.
#
# Byte-exact verification against the HF API is part of the job: a truncated
# 2-bit file loads as garbage rather than erroring, so the size check is the
# only trustworthy gate.
#
# Files go to /kaggle/tmp/escha (ephemeral, big) — NOT /kaggle/working, whose
# 19.5 GB cap and output-versioning would choke on 13 GB.

import importlib.util
import os
import subprocess
import sys

WORK = "/kaggle/tmp/escha"
LOG = f"{WORK}/download.log"
REPO = "aj9o9/Qwen3.8-27B-Escha-W2-GGUF"
FILES = {  # exact bytes from the HF API (?blobs=true) — verified 2026-10-09
    "Escha-Qwen3.8-27B-W2-Q8E.gguf": 10_307_703_008,
    "Escha-Qwen3.8-27B-W2-MTP-F16-headQ4.gguf": 2_926_418_048,
}
SCRIPT = f"{WORK}/download_inner.py"

os.makedirs(WORK, exist_ok=True)
inner = f'''
import os
import time
from huggingface_hub import hf_hub_download
ok = True
for name, want in {FILES!r}.items():
    for attempt in (1, 2, 3):
        try:
            p = hf_hub_download(repo_id="{REPO}", filename=name,
                                local_dir="{WORK}", force_download=(attempt == 3))
            got = os.path.getsize(p)
            print(f"{{name}}: {{got:,}} / {{want:,}} bytes", flush=True)
            if got != want:
                print(f"SIZE MISMATCH {{name}}", flush=True); ok = False
            break
        except Exception as e:
            print(f"attempt {{attempt}} failed: {{e!r}}", flush=True)
            time.sleep(10)
    else:
        ok = False
open("{LOG}", "a").write("DOWNLOAD_OK\\n" if ok else "DOWNLOAD_FAILED\\n")
'''
open(SCRIPT, "w").write(inner)

# Separate kill/launch (HANDOFF rule).
subprocess.run(["pkill", "-f", "download_inner.py"], capture_output=True)
# hf_transfer speeds Kaggle→HF up 3-5x but its env flag must NOT be set
# without the package — huggingface_hub hard-errors instead of falling back.
env = dict(os.environ)
env["HF_HUB_ENABLE_HF_TRANSFER"] = \
    "1" if importlib.util.find_spec("hf_transfer") else "0"
with open(LOG, "w") as fh:
    subprocess.Popen([sys.executable, SCRIPT], stdout=fh, stderr=subprocess.STDOUT,
                     env=env, start_new_session=True)

print(f"DOWNLOAD_RUNNING — poll with cell_03b_status.sh (log {LOG})")