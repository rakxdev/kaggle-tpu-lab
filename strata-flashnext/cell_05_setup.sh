#!/bin/sh
# CELL 5 — setup (the long one, ~20-40 min): ready-made sm_75 engine
# + MTP draft + prepare, in the background with a live log. models-dir is
# /kaggle/working (separate 20 GB, persists across restarts) so the scratch
# quota — already ~55.2/57.6 GiB from the model — never grows. NOTE: setup names
# the data dir Strata-data (capital S) beside the checkout; the flag must match.
# re-run cell_05b_status.sh any time.
cd /kaggle/working/Strata || exit 1
# Kaggle scratch has no room for the 23 GB experts.bin (model 55.2 GiB of the
# ~60 GiB ephemeral ceiling — hit live). Strata's --low-ram mmap mode reads
# the experts straight from the GGUF files instead; this env-guard turns
# setup.py's unconditional --experts-bin step (setup.py line ~3516) into a
# no-op here. Idempotent; also applied at launch via the env var below.
python3 - <<'PYEOF'
src = open("setup.py").read()
old = 'if low_ram and not (pack / "experts.bin").exists():'
if 'STRATA_SKIP_EXPERTS_BIN' in src:
    print("setup.py: guard already present")
elif old in src:
    open("setup.py", "w").write(src.replace(old,
        'if low_ram and not (pack / "experts.bin").exists() and '
        'os.environ.get("STRATA_SKIP_EXPERTS_BIN") != "1":  # Kaggle: no scratch room; --mmap-experts reads the GGUFs'))
    print("setup.py: --experts-bin step env-guarded")
else:
    print("!! setup.py drifted - paste me this")
PYEOF
# PATH prefix: setup.sh picks interpreters by PATH; the apt python (venv-capable)
# must win over /usr/local's ensurepip-less 3.13. With cell 2's .venv in place
# this is belt-and-braces only.
PATH=/usr/bin:$PATH STRATA_SKIP_EXPERTS_BIN=1 nohup ./setup.sh --setup --yes \
  --family coder --model IQ1_M --low-ram mmap \
  --gguf-dir /kaggle/tmp/gguf-coder/IQ1_M \
  --models-dir /kaggle/working/Strata-data \
  --gpus 0,1 --port 8080 > /kaggle/working/strata_setup.log 2>&1 &
echo "SETUP LAUNCHED — poll with cell_05b_status.sh"
sleep 8; tail -5 /kaggle/working/strata_setup.log
