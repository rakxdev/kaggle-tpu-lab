#!/bin/sh
# CELL 5 — setup (the long one, ~20-40 min): venv + ready-made sm_75 engine
# + MTP draft + prepare, in the background with a live log. Progress:
# re-run cell_05b_status.sh any time.
cd /kaggle/working/Strata || exit 1
# PATH prefix: setup.sh picks interpreters by PATH; the apt python (venv-capable)
# must win over /usr/local's ensurepip-less 3.13. With cell 2's .venv in place
# this is belt-and-braces only.
PATH=/usr/bin:$PATH nohup ./setup.sh --setup --yes \
  --family coder --model IQ1_M \
  --gguf-dir /kaggle/tmp/gguf-coder \
  --models-dir /kaggle/tmp/strata-data \
  --gpus 0,1 --port 8080 > /kaggle/working/strata_setup.log 2>&1 &
echo "SETUP LAUNCHED — poll with cell_05b_status.sh"
sleep 8; tail -5 /kaggle/working/strata_setup.log
