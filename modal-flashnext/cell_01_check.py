# CELL 1 — verify Modal auth, create the weights Volume, print the burn-rate.
# Keep the NOTEBOOK itself on CPU / small RAM — the A100 attaches to the
# endpoint app (cell 4), not to this notebook. Idle notebook = $0.
#
# Cost facts (modal.com/pricing): A100-80GB $2.50/h · CPU $0.047/core/h ·
# RAM $0.008/GiB/h · Volume storage free up to 1 TiB/month on Starter.

import subprocess

import modal  # preinstalled in Modal Notebooks; if this line fails: !pip install -U modal

print("modal client:", modal.__version__)

who = subprocess.run(["modal", "token", "whoami"], capture_output=True, text=True)
print(who.stdout.strip() or who.stderr.strip())
if who.returncode != 0:
    raise SystemExit("Modal CLI not authenticated — run a cell with: !modal setup")

vol = modal.Volume.from_name("flashnext-weights", create_if_missing=True)
print("VOLUME_OK flashnext-weights")
print("BURN: endpoint awake ≈ $2.98/h (GPU 2.50 + 4 CPU 0.19 + 48GiB RAM 0.38); $0 when scaled to zero")
