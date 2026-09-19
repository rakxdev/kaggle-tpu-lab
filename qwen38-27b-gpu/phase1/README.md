# qwen38-27b-gpu — Phase 1 (public OpenAI endpoint)

Serves the Phase 0 checkpoint over vLLM's own OpenAI server, API-key
authenticated, behind a cloudflared quick tunnel. Point any OpenAI-compatible
tool at it.

The endpoint is **OpenAI-format only** (`/v1/chat/completions`, `/v1/models`).
It does not speak the Anthropic `/v1/messages` API — that is the GLM kernel's
trick, not vLLM's. Claude Code can still use it via its OpenAI-compatible
provider support.

## Cell (after `PHASE 0B-GPU READY`)

```python
!cd /kaggle/working/kaggle-tpu-lab && git fetch -q origin && git reset -q --hard origin/main && python qwen38-27b-gpu/phase1/serve.py
```

The last printed lines are the endpoint, the API key and the model name. This
cell *is* the server — it stays running until the keepalive expires.

Tunables (env vars): `QWEN_GPU_CTX` (32768), `QWEN_GPU_KEEPALIVE_MIN` (480),
`QWEN_GPU_TOOL_PARSER` (hermes).

## What the run does

1. **Preflight** — checkpoint present, 2 T4s visible.
2. **Starts vLLM** — `vllm.entrypoints.openai.api_server`, TP=2, ctx 32768,
   `--kv-cache-dtype float16`, MTP k=3, `--api-key`. Full output goes to
   `/kaggle/working/vllm_serve.log`; if the server dies, its last 25 lines are
   printed inline so the root cause is visible.
3. **Self-test** — one chat completion.
4. **Tunnel** — cloudflared quick tunnel, 3 attempts, then a background probe
   that confirms the public URL actually answers and replaces it if not.
5. **Ready** — prints the endpoint, key, model and copy-paste env lines, then
   serves until the keepalive.

## Measured on Kaggle 2x T4

- 42 tok/s decode single-stream (MTP), 257 tok/s prefill
- cold load + compile 13.1 min; the compile cache under
  `/root/.cache/vllm` is reused, so a same-config restart is faster
- context ceiling ~48k at these settings (~70k fully squeezed); see the Phase 0
  README. `QWEN_GPU_CTX` must stay inside that budget

## Tool calling

Structured tool calls use `--tool-call-parser hermes`. That is the parser vLLM's
own docs prescribe for Qwen ("the chat template already includes Hermes-style
tool use … you can use the hermes parser to enable tool calls for Qwen models")
and the one Qwen's docs show for Qwen3. If the parser name is rejected at
startup, the server is relaunched once without it — tool calls then arrive as
plain text and the run still serves. Which mode is active is logged.

Known upstream issue: **vLLM #31871** — with the hermes parser, *streaming*
responses can return the tool call as raw text rather than a parsed
`tool_calls` object. Non-streaming requests parse correctly. If your tool
misbehaves on streaming tool calls, disable streaming client-side.

## Good to know

- **API key** is generated per run (`qwen-<hex>`); every run gets a new one and
  a new tunnel URL.
- **Concurrency** is capped at 4 sequences by `--max-num-seqs` (the MTP
  acceptance bug at batch ≥ 4, vLLM issue #55533) and by the ~1.5 GiB/GPU KV
  budget.
- **Tunnel on this image**: cloudflared segfaulted (rc -11) on the *TPU* image;
  the GPU image is different and this is its first real test. The retry + probe
  handles a bad registration, and if the tunnel gives nothing the server is
  still reachable inside the kernel on `127.0.0.1:8000`.
- **Sessions**: Kaggle stops a GPU session after 12 h; the keepalive (480 min)
  ends the cell before that so a forgotten run does not burn quota.
