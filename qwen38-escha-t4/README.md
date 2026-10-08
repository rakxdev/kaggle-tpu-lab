# Qwen3.8-27B-Escha-W2 (native 2-bit) on Kaggle T4 ×2

**Serve the native-2-bit dense 27B from a free Kaggle T4×2 session as an
OpenAI-compatible endpoint with MTP speculative decoding, public URL, API key.**

The model: [aj9o9/Qwen3.8-27B-Escha-W2-GGUF](https://huggingface.co/aj9o9/Qwen3.8-27B-Escha-W2-GGUF)
— EschaLabs' Qwen3.8-27B with its native 2-bit escha payload (2.469 bpw)
decoded **in-kernel** by a custom llama.cpp op. Apache-2.0 end to end.

The catch, stated once: **stock llama.cpp cannot load these files.** They need
the porter's fork ([Ajay9o9/llama.cpp-escha](https://github.com/Ajay9o9/llama.cpp-escha),
branch `escha-w2-dense`, op `GGML_OP_ESCHA_MUL_MAT`), built from source — the
fork ships no release binaries. Cell 2 builds it for sm_75 only.

## Will it fit 2× T4? (the arithmetic)

T4 = 15.83 GiB each. Q8E build + MTP draft, draft pinned to GPU1:

| GPU | Contents | GiB |
|---|---|---|
| 0 | W2 weights (10,307,703,008 B) | 9.60 |
| 0 | compute buffers (ub 2048, 27B dense) | ~1.2 |
| 0 | KV q8_0 — 16 full-attn layers × 4 KV heads × 256 dim = 32 KiB/token | 1.0 @ 32 K |
| 0 | GDN (linear-attn) recurrent state — **unknown until first load** | ≤ 1.5 est |
| 0 | CUDA context | ~0.3 |
| 1 | MTP draft (2,926,418,048 B) | 2.73 |

→ **~12.6–13.6 GiB on GPU0 at 32 K context. Fits.** 64 K adds 1 GiB (tight but
likely OK); 131 K+ needs `-sm layer` across both cards, which halves nothing
and slows decode — and is untested with the custom op. The hybrid architecture
is why the KV line is so small: only 16 of 64 layers are full attention.

## Speed: measured vs estimated

The author measured one RTX 3090 (250 W). The T4 has ~1/3 the memory
bandwidth (320 vs 936 GB/s), which is the whole story for raw decode:

| Metric | RTX 3090 (author, measured) | T4 ×2 (measured live on Kaggle) |
|---|---:|---:|
| Prefill pp512 | 700 tok/s (tensor-core path) | **219.84 ± 1.84 tok/s** (llama-bench) |
| Decode, no MTP (tg128) | 24.0 tok/s | **8.22 ± 0.05 tok/s** (llama-bench) |
| Decode, MTP on (`-np 1`, greedy) | 30.1–40.1 tok/s | **14.86 tok/s** (measured live, 74.1% draft acceptance) |
| MTP Speedup | 1.4–1.8x | **1.81x** |

Real hardware verification confirmed:
- MTP acceptance rate is **74.1%** (20 of 27 drafts accepted).
- Memory footprint: **4,951 MiB on GPU 0** + **8,041 MiB on GPU 1**.
- The 2-bit `GGML_OP_ESCHA_MUL_MAT` kernel arithmetic produces exact answers on Turing (sm_75).

## Honest comparison with the kit that already exists

This repo also has [qwen38-27b-gpu](../qwen38-27b-gpu/): the **same base model
as RedHatAI INT4 (W4A16, 19.5 GB) on vLLM 0.28.0, TP=2, measured 42.0 tok/s
decode on this exact hardware**, with continuous batching for many users.

| | This kit (Escha W2, llama.cpp) | qwen38-27b-gpu (INT4, vLLM) |
|---|---|---|
| Download | **13.2 GB** | 19.5 GB |
| Decode, single stream | ~10–14 (est) | **42.0 (measured)** |
| Multi-user | 1 slot with MTP (others queue) | 16 streams, ~900 tok/s aggregate |
| Max context here | ~64 K comfortable (single GPU0) | 32 K default |
| Stack | llama.cpp fork, built on-session (~20–30 min) | vLLM wheel + 13 min compile |

**Pick by intent:** serving people → INT4 kit. Smallest download, fastest
load, longest context on one card, or the 2-bit kernel itself → this kit.

## Cells (run in order)

| Cell | Job | Quota cost |
|---|---|---|
| 0 | `cell_00_deps.sh` — toolchain + download stack (every session, first) | ~1 min |
| 0b | `cell_00b_pull.sh` — **clone / clean-pull the kit repo (every session, second)** | ~0 |
| 1 | `cell_01_report.sh` — GPU, nvcc, disk; the go/no-go facts | ~0 |
| 2 | `cell_02_setup.sh` — clone fork, verify Turing gates in-kernel, heartbeat, start sm_75 build | ~10–25 min |
| 2b | `cell_02b_status.sh` — poll build → `BUILD_OK` | 0 |
| 3 | `cell_03_download.py` + `cell_03b_status.sh` — 13.2 GB, byte-verified | ~10–20 min |
| 4 | `cell_04_smoke.py` — load + one generation on the real flag set → `ESCHA_SMOKE_OK` | ~3 min |
| 5 | `cell_05_bench.sh` — llama-bench pp512/tg128 → `BENCH_OK` | ~5–10 min |
| 6 | `cell_06_serve.sh` — **endpoint**: server + cloudflared + watchdog → `ESCHA_ENDPOINT_OK` | serving |
| 6b | `cell_06b_proof.py` — run from OUTSIDE: auth, correctness, tok/s, streaming → `PROOF_OK` | ~0 |

Every long job is background + log + a poller cell; every cell re-runs safely.

```sh
sh cell_00_deps.sh    # every session, first
sh cell_00b_pull.sh   # every session, second — picks up any fix I pushed
sh cell_01_report.sh
sh cell_02_setup.sh          # then poll 02b
python3 cell_03_download.py  # then poll 03b
python3 cell_04_smoke.py
sh cell_05_bench.sh
sh cell_06_serve.sh          # then, from any machine:
python3 cell_06b_proof.py https://<something>.trycloudflare.com <key>
```

In a Kaggle notebook the same cells run as `!sh /kaggle/working/kaggle-tpu-lab/qwen38-escha-t4/<cell>`
(`!python3` for the `.py` ones) — cell 0b is what turns my pushed fixes into
your session.

## The MTP contract (two quiet speed-killers)

From the model card, and why cell 6 hard-codes them:

- **`-np 1`** — more slots split the verify batch; the gain evaporates. One
  user at a time; additional requests queue in the server.
- **`--temp 0 --top-k 1`** — greedy. At temp 1.0 draft acceptance falls from
  ~3.1 to ~2.2 accepted tokens and the speedup mostly goes with it. Clients
  that send no temperature get the server default — set it explicitly.

Also in the launch line: `-b 2048 -ub 2048` (~10% of prefill), `-fa on`,
`-ctk/-ctv q8_0` (KV at 32 KiB/token), `--spec-draft-device CUDA1`.

## If it fails

- **Cell 2 self-check prints "branch moved"** — the fork restructured. Re-check
  `ggml/src/ggml-cuda/escha-moe.cu` on the branch before building.
- **`BUILD_FAILED`** — read the tail cell 2b prints. Most likely cause on a
  fresh image: missing system package (`cmake`, `g++`); Kaggle's GPU image
  ships both, so a genuine kernel compile error on sm_75 would be new
  information — paste it upstream.
- **`ESCHA_SMOKE_WRONG_ANSWER` or garbage text** — the W2 kernel compiled but
  computes wrong on Turing. Fallback: `ESCHA_NO_MMA=1` on the launch (forces
  the non-tensor-core prefill path; decode path is unaffected). If garbage
  persists, the 2-bit decode itself is Ampere-only in practice — stop, don't
  tune.
- **`llama-server died during load`** — usually VRAM: the GDN state line in the
  fit table is the unknown. Drop `ESCHA_CTX` to 16384 and retry.
- **Tunnel URL missing** — cloudflared was slow; the cell retries for a minute.
  Re-run cell 6 (it kills the old server first — safe).
- **`ERR_NGROK_725`-style bandwidth walls** — not a thing here: Cloudflare
  quick tunnels have no bandwidth meter.

## Sources

- Model card (sizes, MTP flags, 3090 measurements):
  https://huggingface.co/aj9o9/Qwen3.8-27B-Escha-W2-GGUF — read 2026-10-09
- Fork, branch `escha-w2-dense`: kernel `ggml/src/ggml-cuda/escha-moe.cu`
  (Turing gates), `common/arg.cpp` (`--spec-draft-device`, `--spec-type`),
  `common/speculative.cpp` (`draft-mtp` type) — read 2026-10-09
- Base model: EschaLabs/Qwen3.8-27B-Escha-W2 (Apache-2.0)

## Not confirmed

- Everything in the T4 speed column until cells 5/6b fill it.
- GDN recurrent-state footprint (fit table row marked ≤ 1.5 est).
- `-sm layer` multi-GPU with the custom op (untested by the author; this kit
  deliberately avoids needing it).
- Whether `cp.async` in the Turing-gated prefill block compiles for sm_75 or
  the compiler takes its fallback — either way the cell-2 self-check plus the
  smoke test is the arbiter.
