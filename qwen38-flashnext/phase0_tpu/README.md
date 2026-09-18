# Phase 0b — the TPU run (first-ever v5e-8 tok/s for Qwen3.8-Flash-Next)

Prerequisite: Internet ON. **Dataset attachment is optional** — the resolver
uses the `qwen38-flashnext-w4a16-*` datasets when they are mounted and
downloads any missing shards from HuggingFace in parallel otherwise
(~12–15 min for the full 180 GB at 250–300 MB/s; byte-exact completeness is
asserted either way before the loader runs).

## In the TPU session, paste TWO cells

**Cell 1 — machine report** (NO jax import — see warning below):

```python
!head -3 /proc/meminfo
!nproc
!df -h /kaggle/working /kaggle/tmp / 2>/dev/null | head -6
import urllib.request
print("internet:", urllib.request.urlopen("https://huggingface.co", timeout=15).status)
```

**Cell 2 — the relay agent** (same as the CPU session; it lets me drive everything):
copy it from `../phase0/cell_00_relay_agent.py` — run as a **background thread**
version so it does not block later cells.

**Cell 3 — environment + weights** (the only other cell needed):

```python
!cd /kaggle/working && if [ -d kaggle-tpu-lab/.git ]; then git -C kaggle-tpu-lab fetch -q origin && git -C kaggle-tpu-lab reset -q --hard origin/main && git -C kaggle-tpu-lab clean -qfd; else git clone -q https://github.com/rakxdev/kaggle-tpu-lab; fi && python kaggle-tpu-lab/qwen38-flashnext/phase0_tpu/run_all.py
```

This runs, in order (about 15–20 min, installs dominate):

1. `setup_env.py` — torch 2.13 → vllm 0.28.0 → tpu-inference pinned + overlay →
   `jax[tpu] 0.11.0` stack → torchax paired to torch 2.13 → import probe asserting
   **8 TPU devices**
2. `mount_weights.py` — 30 shards (symlinked datasets + HF download for the rest)
   into `/kaggle/tmp/ckpt` + meta files fetched → index sanity check

After the driver prints `PHASE 0B READY`, **I drive the rest through the relay**:
load the model, generation proof, benchmark. Nothing else is pasted by hand.

## Critical: never `import jax` in the notebook kernel

The TPU device (`/dev/vfio/0`) is **exclusive-open** — the first jax runtime to
touch it holds it. Cell 1's `import jax` in the *kernel* process grabs the TPU,
and the driver (a separate process) then fails with
`open(/dev/vfio/0): Device or resource busy` (seen live; documented in Google's
TPU troubleshooting guide). If that happens: **Kernel → Restart** (the VM, all
pip installs and files survive), re-paste the relay cell, re-run Cell 3 —
everything is cached, so the re-run is quick.

## Machine facts already known

- Host RAM 330 GB / 222 vCPUs (user-confirmed) — the 102.5 GB PLE table pins in host RAM
- HBM: 72.1 GB of weights across 8 chips (9.02 GB/chip, ~7 GB/chip headroom)
- Known-good pairing: torch 2.13.0 / vllm 0.28.0 / tpu-inference c824927 / jax[tpu] 0.11.0
