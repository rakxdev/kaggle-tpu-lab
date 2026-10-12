#!/usr/bin/env python3
# CELL 3 — download Index-Translate-9B Q4_K_M + mmproj vision encoder (background).
#
# Total download size: ~6.40 GB (downloads in ~45-90 seconds with hf_transfer):
#   - Index-Translate-9B.Q4_K_M.gguf        5,780,090,304 bytes (main text model)
#   - Index-Translate-9B.mmproj-Q8_0.gguf     620,690,272 bytes (multimodal vision encoder)
#
# Destination: /kaggle/tmp/translate/models (on ephemeral root overlay with >1 TB space).
# Byte verification against exact HF API bytes ensures zero corrupted downloads.

import importlib.util
import os
import subprocess
import sys

WORK = "/kaggle/tmp/translate"
MODELS_DIR = f"{WORK}/models"
LOG = f"{WORK}/download.log"
REPO = "IndexTeam/Index-Translate-9B-GGUF"

FILES = {
    "Index-Translate-9B.Q4_K_M.gguf": 5_780_090_304,
    "Index-Translate-9B.mmproj-Q8_0.gguf": 620_690_272,
}

SCRIPT = f"{WORK}/download_inner.py"
os.makedirs(MODELS_DIR, exist_ok=True)

inner = f'''
import os
import time
from huggingface_hub import hf_hub_download

ok = True
for name, want in {FILES!r}.items():
    print(f"downloading {{name}} (expect {{want:,}} bytes)...", flush=True)
    for attempt in (1, 2, 3):
        try:
            p = hf_hub_download(repo_id="{REPO}", filename=name,
                                local_dir="{MODELS_DIR}",
                                force_download=(attempt == 3))
            got = os.path.getsize(p)
            print(f"{{name}}: {{got:,}} / {{want:,}} bytes", flush=True)
            if got != want:
                print(f"!! SIZE MISMATCH for {{name}} (got {{got}}, want {{want}})", flush=True)
                ok = False
            else:
                print(f"✓ {{name}} verified", flush=True)
            break
        except Exception as e:
            print(f"attempt {{attempt}} failed: {{e!r}}", flush=True)
            time.sleep(5)
    else:
        ok = False

with open("{LOG}", "a") as f:
    f.write("DOWNLOAD_OK\\n" if ok else "DOWNLOAD_FAILED\\n")
'''

with open(SCRIPT, "w") as f:
    f.write(inner)

# Stop previous download if any (HANDOFF rule)
subprocess.run(["pkill", "-f", "download_inner.py"], capture_output=True)

env = dict(os.environ)
if importlib.util.find_spec("hf_transfer"):
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

with open(LOG, "w") as out:
    subprocess.Popen([sys.executable, SCRIPT], stdout=out, stderr=subprocess.STDOUT,
                     env=env, start_new_session=True)

print(f"DOWNLOAD_RUNNING — poll with cell_03b_status.sh (log: {LOG})")
