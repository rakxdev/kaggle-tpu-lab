# qwen38-27b-gpu — Phase 0 (dense Qwen3.8-27B INT4 on Kaggle 2×T4)

Stack: **vLLM 0.28.0** + **RedHatAI/Qwen3.8-27B-INT4** (19.5 GB, W4A16
group-128 + MTP head) at **TP=2** across both T4s. Marlin W4A16 kernels are
SM75-capable (vLLM PR #29901); the MTP head ships in the checkpoint.

Measured quality of the checkpoint (RedHatAI, served via vLLM, 3 seeds):
IFEval 99.67% recovery, MMLU-Pro 98.81%, GSM8K 101.07%, MATH-500 99.52%,
GPQA-D 98.49%, AIME 98.69% — vs the BF16 model.

## Measured on Kaggle 2x T4 (first cold run)

| Step | Result |
|---|---|
| Environment + 19.5 GB checkpoint | 3.3 min |
| Load + compile | 13.1 min (compile cache is written to disk; a warm restart skips it) |
| Generation proof | "The capital of Germany is" -> "Berlin." — correct |
| Decode, single stream | **42.0 tok/s** (MTP speculative decoding active) |
| Prefill | ~9,012 tokens in 35.1 s = 257 tok/s (first run; includes JIT warmup) |

Per-GPU memory at 0.92 utilization: 10.2 GiB weights, ~1.5 GiB KV cache
(14.4 GiB usable). The KV budget is the constraint for serving — the GDN
layers carry a recurrent state rather than a KV cache, so context length is
not the binding limit, but concurrent streams are.

## Cells (run in order, one at a time)

**CELL 1 — machine + GPU report** (nvidia-smi's CUDA line decides the torch
wheel variant — this is the fact that killed the T4 route's twins):

```python
!nvidia-smi
!head -3 /proc/meminfo
!nproc
!df -h /kaggle/working /kaggle/tmp / 2>/dev/null | head -6
import torch
print("preinstalled torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
```

**CELL 2 — relay agent (background)**: same code as the TPU kit
(`../phase0/cell_00_relay_agent.py`), topics unchanged.

**CELL 3 — driver** (env auto-detects the CUDA variant → resolver → verify):

```python
!cd /kaggle/working && if [ -d kaggle-tpu-lab/.git ]; then git -C kaggle-tpu-lab fetch -q origin && git -C kaggle-tpu-lab reset -q --hard origin/main && git -C kaggle-tpu-lab clean -qfd; else git clone -q https://github.com/rakxdev/kaggle-tpu-lab; fi && python kaggle-tpu-lab/qwen38-27b-gpu/phase0/run_all.py
```

**CELL 4 — load, proof, benchmark** (only after `PHASE 0B-GPU READY`):

```python
!cd /kaggle/working/kaggle-tpu-lab && git fetch -q origin && git reset -q --hard origin/main && python qwen38-27b-gpu/phase0/load_and_bench.py
```

## Rules (each paid for — see the TPU kit's HANDOFF.md)

1. No jax, no `import jax` — this route never needs it; one less deadlock class.
2. One cell at a time; a spinning cell means the kernel is busy — new cells queue.
3. Interrupting a cell does not kill child processes; if weird → Kernel →
   Restart (VM + installs + downloaded shards survive; nothing lives in the kernel).
4. pip is single-connection (~40–90 MB/s) — that's the VM, not a bug; the
   19.5 GB model downloads via hf_transfer multi-stream instead.
5. Everything resumable: the resolver skips byte-exact files; pip steps skip
   importable packages.

## Known risks

- **`FP8 KV cache is not supported ... on Tesla T4 (compute capability 7.5)`.
  vLLM 0.28 defaults this architecture's KV cache to fp8 (the checkpoint does
  not ask for it — this is vLLM's own default), and native fp8e4nv needs SM89+.
  The load pins `kv_cache_dtype="float16"`.
- **vLLM's engine core is spawned, not forked.** Every entrypoint must build the
  engine under `if __name__ == "__main__"`; module-level `LLM(...)` is re-executed
  in the spawned child and trips multiprocessing's bootstrapping check.
- vLLM's GDN/FLA Triton kernels on SM75 are the one untested piece — if the
  load fails on them, the fallback is the llama.cpp route (prebuilt SM75
  binary + unsloth GGUF), which has no such dependency.
- MTP acceptance collapses at batch ≥ 4 (vLLM issue #55533) — max_num_seqs is
  capped at 4 and single-stream benchmarks are unaffected.

## Gotchas already paid for

- **`libcudart.so.13: cannot open shared object file`.** Kaggle's T4 image is a
  CUDA 12.8 host (`/usr/local/cuda-12.8`, `LD_LIBRARY_PATH` → its libs), but
  PyPI vllm 0.28.0 is a cu130 build and its `_C` extension has no rpath. Torch
  2.13's cu130 wheels install the CUDA-13 runtime into
  `site-packages/nvidia/cu13/lib/`, which the loader never searches — so
  `import vllm` dies. `cu_env.py` finds that directory and re-execs the process
  with it on `LD_LIBRARY_PATH` (the value is only read at exec time, so an
  in-process `os.environ` change is not enough). It is imported first by every
  entrypoint and is a no-op on a normal CUDA-13 host.
- **`/kaggle/tmp` does not exist on GPU sessions** (TPU images have it). It is a
  plain directory on the ~1 TB root overlay, so the resolver creates it; the
  19.5 GB checkpoint cannot live on the 20 GB `/kaggle/working` loop device.
