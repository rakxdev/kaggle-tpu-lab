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

echo "== 3. Strata .venv via virtualenv (no ensurepip needed) =="
# Kaggle's pythons can't venv: system 3.13 lacks ensurepip, and even the apt
# 3.12 ships a broken one on this image (seen live 2026-10-04 after a clean
# python3.12-venv install). virtualenv seeds pip from its own bundled wheels,
# bypassing ensurepip entirely; Strata's setup.sh adopts any existing
# .venv/bin/python that has pip (setup.sh lines 13-14) and skips its own
# selection loop. Strata requires python >= 3.10; the system 3.13 qualifies.
if [ ! -x Strata/.venv/bin/python ] || ! Strata/.venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf Strata/.venv
  python3 -m pip install -q virtualenv || pip install -q virtualenv
  python3 -m virtualenv -q Strata/.venv || virtualenv -q Strata/.venv
fi
Strata/.venv/bin/python -c 'import sys; assert sys.version_info >= (3, 10); print("venv python:", sys.version.split()[0])' \
  || { echo "!! .venv python bad — paste me this output"; env | grep -i "^PYTHON"; exit 1; }
Strata/.venv/bin/python -m pip --version >/dev/null 2>&1 \
  || { echo "!! .venv has no pip — paste me this output"; exit 1; }
# Kaggle's sitecustomize hook imports wrapt on every interpreter start;
# inside the venv it is absent and prints a scary (but harmless) error on
# every step of setup. One wheel silences it.
Strata/.venv/bin/python -m pip install -q wrapt 2>/dev/null
echo "Strata .venv ready (setup.sh will adopt it as-is)"

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
