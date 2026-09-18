# Phase 0 — CPU-session smoke test (no TPU, no quota)

Goal: prove the whole Qwen3.8-Flash-Next-on-TPU code path works **before** spending
any TPU time: the fork's engine code passes its tests, and its weight loader accepts
the real community W4A16 checkpoint's tensor names.

## How to run — one cell

New Kaggle notebook → Session options: **Accelerator = None (CPU)**, **Internet = ON**
→ paste this single cell and run it (20–30 min first run):

```python
!cd /kaggle/working && if [ -d kaggle-tpu-lab/.git ]; then git -C kaggle-tpu-lab fetch -q origin && git -C kaggle-tpu-lab reset -q --hard origin/main && git -C kaggle-tpu-lab clean -qfd; else git clone -q https://github.com/rakxdev/kaggle-tpu-lab; fi && python kaggle-tpu-lab/qwen38-flashnext/phase0/run_all.py
```

(The cell is re-run safe: an existing clone is reset to the latest main instead of re-cloned, so every push I make is picked up by simply re-running it.)

Everything runs in order and is teed to `/kaggle/working/phase0_results.txt`:

1. `cell_01_env.py` — machine report (the RAM line decides the n-gram strategy)
2. `cell_02_install.py` — tpu-inference pinned at `c824927` + the fork's overlay (~10–20 min)
3. `cell_03_tests.py` — the fork's 15 CPU tests (expect `15 passed`)
4. `cell_04_loader_smoke.py` — every real checkpoint tensor name through the fork's loader
5. `cell_05_shard_probe.py` — all shard headers → the exact HBM-vs-host memory split

Send me the whole output (or the results file). What each result means:

| Result | Meaning |
|---|---|
| Cell 3: 15 passed | engine code is sound |
| Cell 4: all tensors mapped | the community W4A16 checkpoint loads without changes |
| Cell 4: mapping gaps | I write a small loader shim, you re-run the cell |
| Cell 5: ~78 GB main / ~102 GB PLE | memory plan holds (HBM-resident + host n-gram table) |

After this: cell-by-cell dataset packaging (also CPU), then the single TPU run.
