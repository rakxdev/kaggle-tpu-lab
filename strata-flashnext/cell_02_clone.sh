#!/bin/sh
# CELL 2 — clone Strata into /kaggle/working (persists across session stops)
# and arm the GPU heartbeat (Kaggle stops idle accelerator sessions; the
# download + setup are ~40 CPU-side minutes — this exact thing killed our TPU
# session on 2026-09-20). The heartbeat touches the GPU every 4 min from a
# short subprocess and stands down once the server is serving.
cd /kaggle/working
if [ -d Strata/.git ]; then
  git -C Strata fetch -q origin && git -C Strata reset -q --hard origin/main
else
  git clone -q https://github.com/Niko1221/Strata
fi
echo "Strata at: $(git -C Strata log --oneline -1 | head -c 60)"

# heartbeat: tiny CUDA op via torch in a subprocess (torch preinstalled on
# Kaggle GPU images); skips itself when the Strata server process is alive
cat > /kaggle/working/gpu_heartbeat.py <<'EOF'
import subprocess, sys, time, os
INTERVAL = int(os.environ.get("HEARTBEAT_S", "240"))
TOUCH = ("import torch; x = torch.ones(64, 64, device='cuda:0') @ "
         "torch.ones(64, 64, device='cuda:0'); torch.cuda.synchronize(); "
         "print('HB', float(x.sum()))")
while True:
    r = subprocess.run(["pgrep", "-f", "strata"], capture_output=True, text=True)
    serving = bool(r.stdout.strip())
    if serving:
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
echo "CELL 2 OK — Strata cloned, heartbeat armed"
