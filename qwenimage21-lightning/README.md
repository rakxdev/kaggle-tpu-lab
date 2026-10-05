# qwenimage21-lightning — Qwen-Image-2.1 on Lightning AI, RTX PRO 6000 96GB

Run the **full uncompressed Qwen-Image-2.1** (33.1 GB BF16) on a rented 96 GB
Blackwell card, benchmark it, and expose it as a token-gated HTTP endpoint.

Everything runs as small notebook cells pasted into a Lightning Studio — **no
repo cloning**. The Studio's own filesystem persists, so the 33.1 GB of
weights survive restarts and you never download them twice.

## Session spec (this is the part that costs money)

| Setting | Value | Why |
|---|---|---|
| Machine | **H200 141GB** | 33.1 GB of weights fit natively — no quant, no offload; ~3× the memory bandwidth of the RTX 6000 class and sm_90 is battle-tested in torch |
| Interruptible | **ON** | cheap; downloads resume and cells re-run, so a reclaim costs minutes |
| Duration | 4 hours | hard cap on the Studio's life; ~$15.28 at $3.82/h of the $30 credit |
| Cost | ~$3.82/h (Interruptible ON) | the active price at Confirm time is the one you pay |

Chosen over the RTX PRO 6000 96GB ($3.26/h) for its 1-minute wait and HBM
bandwidth; chosen over the B200 ($9.86/h) because a 4-hour session there costs
$39.44 — more than the entire credit. Any card ≥48 GB runs this kit unchanged:
cell 1 prints the real card and compute capability, and nothing downstream
assumes a particular arch. (The original RTX-6000-vs-A100 note about native
FP8 turned out to be moot — the diffusers pipeline has no FP8 path for anyone.)

**Run on the live Studio (2026-10-05, H200):** card confirmed `9.0, 143771 MiB`,
torch 2.14.1+cu130, bf16 matmul finite, and the full 33.12 GB checkpoint
pulled and **byte-exact-verified against all seven shards**. The download took
**43 seconds** (Xet high-performance transfer) — the "$0.30-0.50 of download
time" estimate below is pessimistic on this platform by roughly 100×.

Two billing facts shape every decision here:

1. **The Studio auto-sleeps after 10 minutes idle** and billing stops. That is
   the safety net — a forgotten cell is not a $13 run.
2. **Duration is a hard cap.** A public endpoint started in hour 3 dies at
   hour 4. If you want to serve for longer, raise Duration *before* starting
   the Studio, not after.

**Kill switch, at any moment:** `pkill -f qi21_server.py` — stops GPU billing
immediately.

## Why full BF16 here, and not a quant

This is the reason the Lightning route and the Kaggle route are different
animals, so it is worth stating plainly.

The model is three parts: a 7B diffusion transformer (14.2 GB), a Qwen3-VL 8B
text encoder (17.5 GB), and a 0.7B VAE (1.4 GB) — **33.1 GB total** at BF16.

- **On this 96 GB card**: all 33.1 GB sit in VRAM with ~60 GB spare. No
  quantization, no `enable_model_cpu_offload()`, no PCIe shuffling. Maximum
  quality, and the thing that actually costs you nothing.
- **On Kaggle 2×T4** (16 GB each, no NVLink): it does not fit. That route
  needs INT8 ConvRot weights (17.3 GB) *and* has to split components across the
  two cards, because a T4 (sm_75) has no native BF16 and no FP8. Measured
  there: **147 s per 1024×1024 image**.

**Do not "optimise" bfloat16 to float16.** BF16 has FP32's exponent range and
FP16 does not. This model's residual streams reach ~3.4e8, which overflows
FP16 to NaN and yields pure black images — measured by the Unsloth team on
T4 hardware across 4 prompts. `QwenImage21Pipeline.supported_inference_dtypes`
offers only `[bfloat16, float32]` anyway.

**INT8 is a better quant than FP8 if you ever need one** — LPIPS 0.064 vs
0.112 — and it is the T4-compatible one. Not needed on this card.

## Cells (run in order, one at a time)

**First, the venv.** The base Studio image is PEP 668 externally-managed (pip
refuses system installs), ships **no torch and no conda** — but does ship `uv`.
Cell 2 builds `~/qwenimage21/venv` and installs into it. Prefix every later
cell with:

```sh
export PATH="$HOME/qwenimage21/venv/bin:$PATH"
```

Dependencies that are NOT optional, both discovered live: **torchvision**
(the checkpoint's Qwen3VL video processor import-raises without it, even for
pure text-to-image) and **python3.12-dev** (Triton JIT-compiles its CUDA
driver wrapper against `Python.h` on first CUDA launch; the header is missing
from the image). Cell 2 handles both.

| Cell | File | Job | Cost |
|---|---|---|---|
| 1 | `cell_01_report.sh` | GPU/arch probe, RAM, disk, bf16 check | $0 |
| 2 | `cell_02_setup.sh` | diffusers **from git** + transformers ≥5.17 | $0 |
| 3 | `cell_03_download.py` | 33.1 GB snapshot, backgrounded, resumable | ~$0.30–0.50 |
| 3b | `cell_03b_status.sh` | poll cell 3 + byte-exact verify | $0 |
| 4 | `cell_04_smoke.py` | one 1024×1024 image, the go/no-go proof | ~$0.20 |
| 5 | `cell_05_bench.py` | cold vs warm, s/step, steps + resolution sweeps | ~$1–2 |
| 6 | `cell_06_serve.sh` | **token-gated server + deadline watchdog** | ~$0.30 |
| 6b | `cell_06b_tunnel.sh` | ngrok tunnel (**free plan caps bandwidth — see 6bb**) | $0 |
| 6bb | `cell_06bb_tunnel_cf.sh` | **Cloudflare quick tunnel — makes it PUBLIC (preferred)** | $0 |
| 6c | `cell_06c_proof.py` | end-to-end proof from outside the Studio | ~$0.15 |

Run from the Studio terminal. Every `.sh` is directly executable; every `.py`
is `python3 <file>`.

```sh
# cell 1 first — everything downstream assumes what it prints
sh cell_01_report.sh
sh cell_02_setup.sh
python3 cell_03_download.py     # then poll: sh cell_03b_status.sh
```

Long jobs are **background + log + a `NNb` poller**, never a poll loop inside
the launching cell. Success is a named sentinel — `SETUP_DONE`, `DOWNLOADED_OK`,
`SMOKE_OK`, `BENCH_DONE`, `PROOF_DONE` — never "it stopped printing."

**Cells 6 and 6b expose a public endpoint.** Run cell 5 first and read the
numbers. Once 6b runs, anyone with the API key can spend your credit.

## Cell 5 is the one that matters for planning

It measures rather than asserts, so the numbers are your card's numbers:

- **cold vs warm** — capacity planning must use the warm number; the first
  render pays for kernel autotuning and allocator growth.
- **s/step** — the constant that decides where credit goes. Everything is
  `steps × s/step`.
- **step sweep (4/8/20/28/40)** — is 40 steps actually needed on 2.1? This is
  your biggest cost lever and nobody should take it on faith. Every step count
  is saved as a PNG at the same prompt and seed so you can *look* at them.
- **resolution sweep (512→2048)** — prices the jump to the model's native 2K.
- **`COMPILE=1`** — opt-in `torch.compile`, deliberately *after* the plain
  numbers. **It is currently broken upstream** — see below.

`QUICK=1 python3 cell_05_bench.py` runs a single arm for a fast headline.

## The endpoint

`qi21_server.py` — FastAPI, N workers (default 3, one model each), a never-fail
SQLite-WAL job queue, and VRAM-aware claiming. **API-first by design: there is no
web generator.** `/` serves the full API reference page (`qi21_docs.html`,
self-contained, zero dependencies) — that page is the community's human interface.

```
GET  /                       API reference page (the docs, no auth)
GET  /health                 per-worker view (no auth)
GET  /queue                  global queue counts (no auth)
POST /generate               submit -> 202 {job_id, queue_position}; never rejects
GET  /jobs/{id}              status + live queue position + result meta
GET  /jobs/{id}/result       the PNG (24h retention)
POST /jobs/{id}/cancel       cancel while queued
POST /v1/images/generations  OpenAI-shaped sync route (waits <=120s)
```

Request: `{prompt, width, height, steps, cfg, seed, negative_prompt}`.
Response: **OpenAI `/v1/images/generations` shaped** —
`{created, data: [{b64_json, url, revised_prompt}], seed, width, height, steps, seconds, seconds_per_step}`.

We match that shape deliberately. vLLM-Omni serves Qwen-Image-2.1 at exactly
this endpoint and its docs drive it through the **OpenAI Python SDK**
(`client.images.generate(...)`), so a real SDK client works against this
server unmodified. The keys after `data` are additive; an OpenAI client
ignores them, and they are what you read for timing. (vLLM-Omni's own 2.1
support is unmerged and was verified only on GB300 — we match the *convention*,
we do not lean on that implementation.)

Three design decisions worth knowing before you change them:

1. **`asyncio.Lock` around the render.** `QwenImage21Pipeline` is not
   thread-safe — two concurrent calls share module buffers and will corrupt
   each other or blow up VRAM. uvicorn runs **one worker** because the lock is
   per-process; more workers would each load their own 33.1 GB copy.
2. **429, not a hang.** When the lock is held, callers get a fast
   `Retry-After` instead of a connection that looks dead. That is also what
   stops one client walking all over a shared endpoint.
3. **`/v1/images/generations` is an alias, not real OpenAI.** The path matches
   so familiar clients can be pointed at it, but the response schema is this
   server's own and `b64_json` is PNG. A client expecting OpenAI's exact
   shape needs a shim.

The API key is generated in cell 6, held in the environment, and never written
into `qi21_server.py` or committed. Re-running cell 6 issues a new key and
retires the old one.

## Billing guards

- **Deadline watchdog** (cell 6): kills server *and* tunnel after
  `QI21_MAX_MIN` (default 120). A forgotten session cannot drain the credit.
- **Kill switch**: `pkill -f qi21_server.py`.
- **Auto-sleep**: the Studio's own 10-minute idle sleep.
- **pkill and launch are always separate lines** in both shell cells — a
  combined `pkill -f X && nohup X &` can match its own launch text and kill
  the new process (see `HANDOFF.md`).

## If it fails

- **`ImportError: cannot import name 'QwenImage21Pipeline'`** — the PyPI
  diffusers won the install race. `QwenImage21Pipeline` is post-0.40; the
  model card requires `pip install git+https://github.com/huggingface/diffusers`.
  Re-run cell 2.
- **`!! image is essentially flat/black`** — the NaN signature. Means the load
  landed in float16, or FP8 on a card without it. Check cell 1's
  `compute_cap` and cell 4's dtype. Do not run cell 5.
- **`!! no CUDA — the Studio has no GPU attached`** — you started a CPU Studio.
  Switch the machine to the RTX PRO 6000 in the GPU switcher.
- **`NOT READY: SHORT transformer/...`** — a download died mid-shard. Re-run
  cell 3; it resumes rather than restarting.
- **`!! nothing answering on 127.0.0.1:8080`** — the 33.1 GB load takes a
  minute. `tail -20 $HOME/qwenimage21/serve.log` and wait; only re-run cell 6
  if `MODEL_READY` is absent from the log.
- **`!! no public URL in the log yet`** — the tunnel was still starting. The
  cell waits ~20 s; re-run it (it kills the old tunnel first, so it is safe).
- **`!! ngrok rejected the token`** — authtoken pasted wrong. Get a fresh one
  from the ngrok dashboard.
- **Visitors see `ERR_NGROK_725 Network bandwidth exceeded`** — ngrok's free
  plan has a 1 GB/month cap and a 2K PNG is ~9-12 MB, so it runs out fast.
  Switch to `cell_06bb_tunnel_cf.sh`; Cloudflare quick tunnels have no such cap
  and need no account.
- **The public URL changed after a restart** — expected for a Cloudflare quick
  tunnel (the hostname is ephemeral). The docs page resolves its own origin at
  runtime, so it always shows correct examples; just re-share the new URL.
- **Unexpected `500` under load** — read `serve.log` for a CUDA OOM. Lower
  `QI21_MAX_STEPS` or `QI21_MAX_PIXELS`.

## Facts this kit is built on (checked 2026-10-05)

- Qwen-Image-2.1 = 7B single-stream DiT (32 layers) + Qwen3-VL 8B text encoder
  + 16× RGBA autoencoder; 33.1 GB BF16 (14.2 / 17.5 / 1.4). — official repo
- Shard byte sizes in `cell_03_download.py` are read from the HF API tree
  listing of `Qwen/Qwen-Image-2.1`, and sum to 33.13 GB.
- Official example: `QwenImage21Pipeline.from_pretrained(..., torch_dtype=torch.bfloat16)`,
  `num_inference_steps=40`. — [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1)
- `supported_inference_dtypes` is `[bfloat16, float32]` — no fp16 path exists.
- FP16 NaN on a T4 at 4/4 prompts, with residual streams measured at 3.4e8;
  ComfyUI also runs this model in FP32 on T4. —
  [unsloth#12690](https://github.com/unslothai/unsloth/pull/12690)
- INT8 LPIPS 0.064 vs FP8 0.112 (mean, 4 prompts) — INT8 is both the
  T4-compatible and the more faithful quant. —
  [Unsloth Qwen-Image-2.1 docs](https://unsloth.ai/docs/models/qwen-image-2.1)
- Kaggle 2×T4 measured 147 s per 1024×1024 / 40 steps with INT8 ConvRot
  attention, vs 272 s with PyTorch FP32 attention. —
  [Kaggle discussion](https://www.kaggle.com/discussions/general/743701)
- Lightning bills per second, auto-sleeps idle Studios after 10 min, and
  supports exposing arbitrary servers on public ports. —
  [Lightning AI billing](https://lightning.ai/docs/platform/overview/faq/billing)

## Gotchas that will bite you (all confirmed, all current)

- **`torch.compile` is BROKEN for this model.** `diffusers#14821` is an open
  **bug issue** ("Qwen Image 2.1 transformer is incompatible with
  `torch.compile`", opened 2026-09-20, still open), not a PR. There are four
  host-dependent graph breaks, two of them data-dependent per step:
  `use_kv_cache=True` makes step 0 take the "extract" path and later steps the
  "cached" path, and `build_token_metadata` uses `nonzero()` so the index
  vector's *length* depends on mask values — which is exactly the recompile
  trigger. `fullgraph=True` hard-errors; plain compile may still run via graph
  breaks (unconfirmed). `COMPILE=1` in cell 5 is a curiosity, not a speedup.
- **The CFG kwarg is `true_cfg_scale`, and `guidance_scale` does not exist.**
  Passing `guidance_scale=` to `QwenImage21Pipeline` raises `TypeError` — the
  parameter is absent from the signature. The default is `true_cfg_scale=1.0`
  because "Qwen-Image 2.1 is meant to be sampled without guidance"; raising it
  above 1.0 makes the DiT run **twice per step**.
- **`negative_prompt` is ignored when `true_cfg_scale` is not above 1.** Since
  the default is 1.0, a negative prompt is silently a no-op at the default
  setting. The server only forwards it when you set `cfg > 1`.
- **`use_kv_cache=True` by default, and it is not bit-exact in reduced
  precision.** If you ever compare images across precisions, pin this flag.
- **Scheduler is `FlowMatchEulerDiscreteScheduler`.** ComfyUI's native 2.1
  templates use 25 steps / cfg 1 / euler / **simple** — which is where the
  "Euler/simple" in the Kaggle notebook came from.
- **There is no LightX2V distilled variant for 2.1.** Their HF org is entirely
  1.x / 2512 / Edit-2511. Anyone citing a "LightX2V 4-step for 2.1" is wrong.
  Real community 2.1 adapters do exist — `chriswritescode/Turbo8-LoRA-Qwen-Image-2.1`
  (8 step, text exact-match drops 95%→75%), `Viggle/Qwen-Image-2.1-viggle-turbo`
  (6 step), `ThakiCloud/Qwen-Image-2.1-FewStep-v0.1` (5/8 step, T2I only,
  editing "not verified"), `PrunaAI/Pruna-Qwen-Image-2.1` (5/8 step, card
  admits it "does not yet match the visual quality of the base model").
  **Every one of them needs scheduler surgery** (`shift_terminal=None`) to work
  at all, so they are a separate exercise — which is why cell 5's sweep is
  20/28/40 and not 4/8/20. The base model at 4–8 steps is simply bad; those
  step counts only make sense with a distilled adapter.
- **Text rendering is what low steps break first.** The demo prompt is the
  official model card's neon-**sign** prompt, i.e. a text-rendering test, and
  text is exactly what degrades at low step counts with no CFG. Do not
  conclude "20 steps is unusable" from that prompt — re-test a scene prompt.

## Measured on the live session (H200, 2026-10-05)

Full BF16, no offload, `use_kv_cache=True`, CFG 1, Euler FlowMatch. Cell 4 smoke:

```
LOADED in 6.5s — 30.2 GiB resident of 139.8 GiB
GENERATE 6.9s at 40 steps = 0.17 s/step     (1024x1024)
VRAM peak 36.8 GiB
```

Cell 5 bench (warm, same prompt/seed throughout):

| Arm | Steps | Res | Seconds | s/step |
|---|---|---|---|---|
| cold | 40 | 1024² | 7.4 | 0.18 |
| **warm** | **40** | **1024²** | **5.9** | **0.15** |
| sweep | 20 / 28 / 40 | 1024² | 3.0 / 4.2 / 5.9 | 0.15 flat |
| res | 20 | 512² | 0.9 | 0.05 |
| res | 20 | 768² | 1.8 | 0.09 |
| res | 20 | 1312² (sweep asked 1328; resized, must be ÷32) | 5.4 | 0.27 |
| res | 20 | 2048² | 15.8 | 0.79 |

Reading: s/step is **flat across step counts** — steps are a pure quality dial;
1024→2048 costs ~5.3× (4× pixels, slightly superlinear from attention); VRAM at
2K peaks 56.5 GiB of 139.8; cold-vs-warm overhead is only **1.5 s**. At
$3.82/h: **$0.0063 per 1024²/40-step image** (~10 images/min warm), $0.017 per
2K/20-step image — 8–10× cheaper than hosted APIs ($0.053–0.134), and ~25×
faster than the Kaggle 2×T4 INT8 route (147 s).

Endpoint proof (cell 6c, run from a machine outside the Studio, over ngrok):
health 200; no/bad token → 401 both; one real 512²/20-step generation 200 in
2.83 s server-side (4.6 s wall incl. tunnel), PNG 481 KiB, pixels healthy;
4 concurrent requests → all 200, fully serialised, no 429 needed, no corruption.

## Not confirmed

- **Native FP8 does not help the BF16 path.** The card has FP8/FP4 tensor
  cores, but `QwenImage21Pipeline` has **no FP8 path at all** — FP8 exists
  only in third-party weights and vLLM-Omni.
- **Whether 20 steps is visually acceptable on 2.1.** Cell 5 writes a PNG per
  step count at a fixed prompt and seed; that judgement is yours to make by
  looking. (The demo prompt is a text-rendering test — the hardest case for
  low steps.)
- **Whether non-`fullgraph` compile works at all** (graph breaks instead of a
  hard error). The 2.1 docs page *does* recommend `pipe.transformer.compile()`
  and `QwenImage21FlexAttnProcessor` "once the model is compiled", which
  contradicts the open bug. `COMPILE=1` settles it on your build.

## License

The model is under the **Qwen Research License — non-commercial use only**.
This kit's code is in this repo; the weights are not redistributed and are
fetched from `Qwen/Qwen-Image-2.1` at run time. Read the model license before
any use you care about.
