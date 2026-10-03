#!/bin/sh
# CELL 2 — session prep. Four jobs in the order the session needs them:
#   1. clean-pull OUR kit repo (fetch + hard reset + delete untracked files):
#      every driver fix lands as a clean tree; nothing outside the repo dir
#      is touched (Strata/, strata_api_key, logs, /tmp tokens all survive)
#   2. clone/update Strata
#   3. make a venv-capable python and pre-create Strata's .venv — Kaggle's
#      default python3 is 3.13 WITHOUT ensurepip, and Strata's own recovery
#      (setup.sh installs python3-venv to /usr/bin) then still PATH-finds the
#      broken interpreter and exits; pre-creating .venv skips that selection
#      loop entirely (source: setup.sh lines 8-36,
#      https://github.com/Niko1221/Strata/blob/main/setup.sh)
#   4. arm the GPU heartbeat (Kaggle stops idle accelerator sessions; the
#      download + setup are ~40 CPU-side minutes)
cd /kaggle/working || exit 1

echo "== 1. kit clean-pull =="
if [ -d kaggle-tpu-lab/.git ]; then
  git -C kaggle-tpu-lab fetch -q origin
  git -C kaggle-tpu-lab reset -q --hard origin/main
  git -C kaggle-tpu-lab clean -qfd
else
  git clone -q https://github.com/rakxdev/kaggle-tpu-lab
fi
echo "kit:    $(git -C kaggle-tpu-lab log --oneline -1 | head -c 70)"

echo "== 2. Strata =="
if [ -d Strata/.git ]; then
  git -C Strata fetch -q origin && git -C Strata reset -q --hard origin/main
else
  git clone -q https://github.com/Niko1221/Strata
fi
echo "strata: $(git -C Strata log --oneline -1 | head -c 70)"

echo "== 3. venv-capable python =="
if ! /usr/bin/python3 -c 'import sys, venv, ensurepip; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  echo "installing python3-venv via apt (one-time, ~1 min)..."
  apt-get install -y -qq python3 python3-venv python3-pip 2>/dev/null \
    || sudo apt-get install -y -qq python3 python3-venv python3-pip
fi
/usr/bin/python3 -c 'import sys, venv, ensurepip; print("venv-capable python:", sys.version.split()[0])' \
  || { echo "!! /usr/bin/python3 still not venv-capable — paste me this output"; exit 1; }
if [ ! -x Strata/.venv/bin/python ]; then
  /usr/bin/python3 -m venv Strata/.venv \
    && Strata/.venv/bin/python -m pip --version >/dev/null 2>&1 \
    && echo "Strata .venv ready (setup.sh will adopt it as-is)" \
    || { echo "!! .venv creation failed — paste me this output"; exit 1; }
else
  echo "Strata .venv already present — keeping it"
fi

echo "== 4. GPU heartbeat =="
cat > /kaggle/working/gpu_heartbeat.py <<'EOF'
import subprocess, sys, time, os
INTERVAL = int(os.environ.get("HEARTBEAT_S", "240"))
TOUCH = ("import torch; x = torch.ones(64, 64, device='cuda:0') @ "
         "torch.ones(64, 64, device='cuda:0'); torch.cuda.synchronize(); "
         "print('HB', float(x.sum()))")
while True:
    r = subprocess.run(["pgrep", "-f", "strata"], capture_output=True, text=True)
    if r.stdout.strip():
        print(time.strftime("[%H:%M:%S]"), "heartbeat: server running — skip", flush=True)
    else:
        try:
            h = subprocess.run([sys.executable, "-c", TOUCH], capture_output=True,
                               text=True, timeout=180)
            print(time.strftime("[%H:%M:%S]"),
                  "heartbeat: GPU touched" if "HB" in (h.stdout or "")
                  else f"heartbeat: failed {(h.stderr or '').strip()[-100:]}",
                  flush=True)
        except Exception as e:
            print(time.strftime("[%H:%M:%S]"), f"heartbeat: error {e!r}", flush=True)
    time.sleep(INTERVAL)
EOF
nohup python3 /kaggle/working/gpu_heartbeat.py > /kaggle/working/heartbeat.log 2>&1 &
sleep 3; tail -2 /kaggle/working/heartbeat.log
echo "CELL 2 OK — kit clean-pulled, Strata ready, .venv built, heartbeat armed"
