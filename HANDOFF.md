# HANDOFF — Qwen3.8-Flash-Next on Kaggle TPU v5e-8

Written: **2026-09-19, 05:43 IST (Saturday)**, end of a ~18-hour working session.
Repo: `github.com/rakxdev/kaggle-tpu-lab` (branch `main`, HEAD `dbccc82`).
Read this top to bottom before touching anything.

## 1. Context — what this project is

Serve **Qwen3.8-Flash-Next** (125B-A6B MoE + 51B n-gram PLE + 4B MTP) on
Kaggle's free TPU v5e-8 (8×16 GB HBM) via the **DQN-Labs/nexus-tpu-fork**
(JAX/TPU overlay on vllm's tpu-inference), as a new recipe
`qwen38-flashnext/` in this repo. Weights: community W4A16
(`VnimanieAI/Qwen3.8-Flash-Next-W4A16`, compressed-tensors group-128) — chosen
because the fork's `quant.py` does not load NVFP4 (its own docstring).
Quality claim: ~99% of FP8 for NVFP4 analogs; W4A16 unmeasured — Phase 3 evals
will measure what we serve.

Hardware/memory plan (verified): 72.1 GB weights HBM-resident (9.02 GB/chip,
~7 GB/chip headroom) + 102.5 GB PLE n-gram table pinned in host RAM. TPU VM
confirmed **~396 GB RAM, 96 vCPU** (live reading), scratch overlay ~1 TB free.

## 2. What is DONE (verified, do not redo)

| Item | Evidence |
|---|---|
| 13 public datasets `rakeshbehera42/qwen38-flashnext-w4a16-01..13` | `verify_datasets.py` → "ALL VERIFIED — sizes match, all public" (byte-exact vs HF) |
| Working env recipe (the only order that works) | torch 2.13.0 base → vllm 0.28.0 → tpu-inference `c824927` with `--no-deps` → explicit `jax[tpu] 0.11.0` + `flax 0.12.8` + `tpu-info 0.7.1` + torchax `0.0.14.dev20260918` (`--no-deps`) |
| Fork import chain on TPU | live: `torch 2.13.0+cu130 \| vllm 0.28.0 \| jax 0.11.0`, **8× TpuDevice**, `STEP 1 OK` |
| Fork CPU tests | 16 passed, 1 skipped |
| Loader audit | all 222,842 checkpoint tensors map through the fork's loader; 0 unmapped |
| One-cell runners + relay agent + verifier | all committed (`qwen38-flashnext/phase0*/`) |
| (superseded but done) GLM/Qwen tunnel fix | commit `cdc7560` — retries + reachability probe |

## 3. CURRENT STATE at handoff time

- The TPU session running Phase 0b was **accidentally stopped at ~05:15 IST**
  (Stop-session misclick) while the checkpoint resolver was mid-download.
  Consequence: the VM, pip installs, and partially downloaded shards are gone.
- The notebook was **re-started and is queued again (position ~155)** on the
  second account.
- **Nothing of value was lost** except the partial download: the 13 datasets
  hold everything, and the resolver re-downloads only what datasets don't cover.

## 4. Blockers (each with owner + next action)

1. **TPU slot (queue position ~155, second account)** — owner: user — next:
   keep notebook #2 queued; attach the 13 datasets NOW while queued (Add Input
   → search `qwen38-flashnext-w4a16`). Attachment is optional (resolver
   downloads otherwise, ~12–15 min) but saves that time every run.
2. **Second queue ticket** — owner: user — optional: start a TPU session on the
   FIRST account (`rakeshbehera42`, ~17 h quota left) in parallel; datasets are
   its own, attach natively. Whichever account lands first runs Phase 0b.
3. **Fork's GDN numerics unvalidated vs GPU reference** (fork docs' own gap) —
   owner: assistant — next: Cell 4's generation proof decides. Coherent English
   → proceed; gibberish → the JAX port has a numerics bug → escalate to the
   fork's issues or run plan-B (own engine port; `phase0_tpu` mapping doc exists).
4. **Kaggle dataset-attach UI slowness** — owner: user — attach one dataset at a
   time; the resolver tolerates any partial set.

## 5. NEXT STEPS (exact, in order)

When a TPU session lands (either account):

1. **Cell 1 — machine report (NEVER `import jax` in the kernel — see §7):**
   ```python
   !head -3 /proc/meminfo
   !nproc
   !df -h /kaggle/working /kaggle/tmp / 2>/dev/null | head -6
   import urllib.request
   print("internet:", urllib.request.urlopen("https://huggingface.co", timeout=15).status)
   ```
2. **Cell 2 — relay agent (background-thread version)** from
   `qwen38-flashnext/phase0/cell_00_relay_agent.py` (topics
   `ktl-cmd-41cbd7210e5b` / `ktl-out-f28de123fc89`). If running sessions in
   parallel, generate a FRESH topic pair per session or commands collide.
3. **Cell 3 — driver:**
   ```python
   !cd /kaggle/working && if [ -d kaggle-tpu-lab/.git ]; then git -C kaggle-tpu-lab fetch -q origin && git -C kaggle-tpu-lab reset -q --hard origin/main && git -C kaggle-tpu-lab clean -qfd; else git clone -q https://github.com/rakxdev/kaggle-tpu-lab; fi && python kaggle-tpu-lab/qwen38-flashnext/phase0_tpu/run_all.py
   ```
   Runs `setup_env.py` (~15 min first run) then `mount_weights.py` (instant with
   datasets / ~12–15 min download without). Ends `PHASE 0B READY`.
4. **Cell 4 — `load_and_bench.py`** (same self-update pattern):
   load TP=8 → greedy proof ("The capital of Germany is" → coherent) → decode
   tok/s → prefill tok/s. Send ALL output back.
5. After numbers: wire the serving recipe (`launch.py serve --model
   qwen38-flashnext`), tunnel reuse, eval harness (Phase 1–3 per the phase plan).

## 6. Access & credentials (pointers only — no secrets in this file)

- Kaggle token (rakeshbehera42): `~/.kaggle/access_token` on the local machine
  and on relay VMs. Second account's token: not yet provided.
- GitHub push to `rakxdev/kaggle-tpu-lab`: user's fine-grained PAT, used
  per-push, never stored.
- HF token: optional (download rate limits are request-count based — 3,000
  resolvers/5 min anonymous — the whole checkpoint is ~40 requests).
- ntfy relay topics: `ktl-cmd-41cbd7210e5b` / `ktl-out-f28de123fc89` (private,
  session-bound; rotate per session).

## 7. Hard-won rules (each paid for today — do not relearn them)

1. `/kaggle/working` is a **20 GB loop device**; stage anything big in
   `/kaggle/tmp` (~1 TB). This caused every failed dataset upload.
2. `pkill -f PATTERN` dies if PATTERN appears anywhere in the same compound
   command's launch text — kill and launch must be **separate** commands.
3. pip's resolver cannot co-install vllm 0.28.0 with tpu-inference@pin
   (numba 0.65 vs 0.62) — install vllm first, tpu-inference with `--no-deps`.
4. The VM's preinstalled torchax targets old torch — pair torch 2.13.0 with
   torchax `0.0.14.dev20260918` via `--no-deps`.
5. `/dev/vfio/0` is exclusive-open: **never `import jax` in the notebook
   kernel** (preinstalled jax grabs the TPU; the driver process then fails
   "Device or resource busy" — Google-documented). Kernel → Restart releases it.
6. `os.fork()` in a JAX-threaded kernel deadlocks (JAX #21073, CPython #100228)
   — no `!`-shell cells after jax is imported; interrupting a cell does not
   kill its child processes.
7. ntfy.sh rate-limits anonymous posters (429) — the relay agent uses a
   blocking stream + Retry-After backoff + persistent seen-state.
8. HF `/raw/` returns Git-LFS pointers — use `/resolve/` for LFS files
   (the 22 MB index.json is LFS).
9. Kaggle's HF-side: never trust "created" until `verify_datasets.py` passes;
   sizes are byte-checked against the HF manifest.

## 8. Reference links

- Fork: `github.com/DQN-Labs/nexus-tpu-fork` (docs/qwen4_exp.md = architecture map + plan-B porting guide)
- Checkpoint: `huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16` (NVFP4 alt: `nvidia/Qwen3.8-Flash-Next-NVFP4` — needs a fork dequant path, Phase 4)
- Official CUDA recipe: `recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next`
- Bandwidth probe / packer / verifier: `qwen38-flashnext/phase0/`
- Phase plan: Phase 0a ✅ datasets → **Phase 0b (this handoff's subject)** → Phase 1 recipe → Phase 2 hardening → Phase 3 evals → Phase 4 MTP/vision/NVFP4
