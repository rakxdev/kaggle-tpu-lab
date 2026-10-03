# modal-flashnext — Qwen3.8-Flash-Next endpoint on Modal, A100-80GB, $30 budget

Serve **Qwen3.8-Flash-Next** (125B/6B-active MoE) as an OpenAI-compatible public
endpoint from Modal Notebooks. Everything runs as inline notebook cells — **no
repo cloning**. The endpoint is a deployed Modal app with scale-to-zero, so the
$30 credit is only drained while the A100 is actually awake.

## Why this shape (researched, source-cited)

| Decision | Choice | Why |
|---|---|---|
| Engine | llama.cpp (official `server-cuda` image, pinned b11379) | day-0 `qwen4exp` support (PR #27742); GSQ-RCO quants load in vanilla llama.cpp; no compile step |
| Weights | ISTA-DASLab GSQ-RCO **IQ3_S** (83.6 GB: 54.8 weights + 28.8 n-gram) | largest quant whose weights fit one A100-80; scores at/above BF16 base on AIME25 (100.0), GPQA-D (92.93 vs 91.92), task avg 93.26 vs 93.12 |
| n-gram shard | mmap'd from Volume, page-cached in 48 GiB host RAM | card: "Keeping the n-gram table in RAM rather than on disk removes the paging cost entirely"; weights can't do this — VRAM bandwidth 2 TB/s vs RAM ~0.1 TB/s |
| MTP | quimmedes `mtp-*Q4_K_M.gguf`, `--spec-type draft-mtp` | speculative decoding for single-user latency |
| Hosting | Modal `@app.server`, `min_containers=0`, `--api-key` gate | per-second billing, scale-to-zero, public URL — no ngrok needed |
| Not used | Strata (streams experts to save VRAM — pointless on 80 GB), NVIDIA NVFP4 checkpoints (Blackwell-only, multi-GPU, ~$28-48/h) | |

## Cost (verified 2026-10 against modal.com/pricing)

| Item | Rate | While endpoint awake |
|---|---|---|
| A100-80GB | $2.50/h | $2.50 |
| CPU 4 cores | $0.047/core-h | $0.19 |
| RAM 48 GiB | $0.008/GiB-h | $0.38 |
| **Total** | | **≈ $2.98/h**, $0 scaled to zero |
| Volume (~87 GB) | first 1 TiB/month free | $0 |

Validation run ≈ $5-8 of the $30. `!modal app stop flashnext-endpoint` kills all billing.

## Cells (paste in order, one at a time — every file is a ready-to-paste cell)

| Cell | File | What it does | Cost |
|---|---|---|---|
| 1 | `cell_01_check.py` | auth check, create Volume `flashnext-weights`, burn-rate | $0 |
| 2 | `cell_02_download.py` | writes + runs the downloader: 87 GB into the Volume (resumable) | cents, 5-15 min |
| 3 | `cell_03_probe.py` | verifies every CLI flag exists in the pinned build | cents |
| 4a/4b/4c | `cell_04_app.py` | three marked sections: API key env → write app → `modal deploy`, prints public URL (or paste the whole file as one cell) | $0 until first request |
| 5 | `cell_05_smoke.py` | health, models, first chat, warm tok/s | ~$0.20 (cold start billed) |

`flashnext_app.py` in this folder is the canonical app; cell 4b embeds the same
content (verified byte-identical at build time). Edit only two things: run cell
4a to set `APP_API_KEY`, and set `URL` in cell 5 from the deploy output.

## If it fails

- **cell 3 `from_registry` "manifest unknown"** — the image tag naming changed;
  use `ghcr.io/ggml-org/llama.cpp:server-cuda` (unpinned) and re-run.
- **cell 3 prints `MISS` for any flag** — llama.cpp renamed it. Paste the full
  probe output; the app file's `cmd` gets the corrected flag (no guessing).
- **cell 5 hangs on 503** — cold start still loading (llama-server returns 503
  until weights are in VRAM); the client in cell 5 tolerates it via long timeout.
  If startup_timeout is exceeded, raise `20 * MINUTES` in the app file.
- **CUDA error on A100 at first load** — the official image didn't cover sm_80
  (unexpected; A100 is Modal's most common GPU). Fallback: build llama.cpp from
  source in the image with `-DCMAKE_CUDA_ARCHITECTURES=80` (adds ~10 min image
  build, cached afterward).
- **Out-of-memory in VRAM** — drop `--mmproj` (0.91 GB) or lower `-c` to 16384.

## Sources

- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF model card (shard table, memory
  requirements, IQ3_S benchmarks, llama.cpp usage incl. `-lm mmap --lazy-mode on`)
- ggml-org/llama.cpp PR #27742 (qwen4exp), docs/docker.md (image tags), releases (b11379)
- Modal docs: examples `liquidai_embeddings_server` (llama.cpp + `@app.server`
  pattern), `very_large_models` (Volume caching, scale-to-zero), guide/notebooks
  (CLI + auth inside Modal Notebooks), pricing page, `modal.web_server` reference
- quimmedes/Qwen3.8-Flash-Next-MTP-GGUF (MTP head + draft-mtp flags)
