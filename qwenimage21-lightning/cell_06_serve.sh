#!/bin/sh
# CELL 6 — start the token-gated server, with a deadline watchdog.
#
# *** THIS CELL MAKES A PUBLIC ENDPOINT. Run cell 5 first and look at the
# *** numbers. Once you run this, anyone with the token can spend your credit.
#
# Cost: the server holds the GPU awake for as long as it runs. At ~$3.26/h
# that is the single most expensive thing in the kit. The watchdog below caps
# it at QI21_MAX_MIN (default 120) so a forgotten session cannot drain the
# rest of your 30 credits. Set QI21_MAX_MIN=60 for a tighter cap.
#
# Safe to re-run: it kills the old server and watchdog first, then launches
# fresh, so you never end up with two 33 GB copies fighting for VRAM.

WORK="${QI21_WORK:-$HOME/qwenimage21}"
MAX_MIN="${QI21_MAX_MIN:-120}"
LOG="$WORK/serve.log"
WATCH="$WORK/watchdog.sh"

mkdir -p "$WORK"
cd "$WORK" || exit 1

# ---- 0. the server file ---------------------------------------------------
# This kit does NOT clone the repo, so qi21_server.py is not on the Studio's
# disk. It is embedded here verbatim (the copy beside this cell in the repo is
# the canonical one you read and edit; this is the same bytes). The heredoc
# is quoted so nothing is expanded at write time.
if [ -f qi21_server.py ] && grep -q QI21_API_KEY qi21_server.py 2>/dev/null; then
  echo "server file already present — keeping it"
else
  cat > qi21_server.py <<'QISERVER_EOF'
#!/usr/bin/env python3
"""Qwen-Image-2.1 HTTP server — full BF16 on a 96 GB card, one model in VRAM.

Written by cell_06_serve.sh (which is the canonical copy — this file is here
so you can read/edit the server without digging through a heredoc).

DESIGN NOTES — the three decisions that matter:

1. **One request at a time, enforced by a lock.** This is the single most
   important line in the file. `QwenImage21Pipeline` is NOT thread-safe: two
   concurrent `.()` calls share one set of module buffers and will corrupt
   each other's latents or blow up VRAM. `asyncio.Lock` turns a pile of
   simultaneous requests into an orderly queue instead. uvicorn is started
   with ONE worker (cell 6) because a lock is per-process and would not
   protect anything if there were several.

   The mechanism is the scheduler, not a guess: `FlowMatchEulerDiscreteScheduler
   .set_timesteps` mutates `self.timesteps`/`self.sigmas`, so two interleaved
   calls corrupt each other's denoising schedule. The same class of bug was
   hit and fixed for StableDiffusionPipeline in diffusers#3672 (an IndexError
   from `self.alphas_cumprod[timestep]` inside the scheduler under two
   threads). There is no official diffusers statement about lock-vs-not for
   QwenImage21Pipeline specifically — this rests on #3672 plus that scheduler
   mutability, so treat it as strongly evidenced rather than documented.
   `enable_model_cpu_offload()` would be doubly unsafe (it mutates module
   .to()/.cpu() state per call); on 96 GB we never need it.

2. **429, not a hang.** When the lock is held the caller gets a fast, honest
   "busy, retry shortly" with a Retry-After, instead of a connection that
   sits open for minutes and looks like a dead server. That is also what
   keeps a shared public endpoint from being walked all over by one client.

3. **Base64 PNG in, base64 PNG out.** One POST -> one JSON response, so it
   works from curl without a multipart parser and from any HTTP client. The
   `/v1/images/generations` path is a deliberate alias of `/generate` so
   OpenAI-shaped clients can be pointed at it, but note this is NOT a real
   OpenAI endpoint: the response body is our own shape and `b64_json` is PNG
   data, so a client expecting OpenAI's exact schema will need a shim.

Auth is a bearer token read from the environment (never from this file, never
committed). The weights come from the Studio-local snapshot that cell 3
downloaded — no download at serve time, no HuggingFace token needed.
"""

import asyncio
import base64
import io
import os
import pathlib
import time

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

WORK = pathlib.Path(os.environ.get("QI21_WORK", pathlib.Path.home() / "qwenimage21"))
PORT = int(os.environ.get("QI21_PORT", "8080"))
TOKEN = os.environ.get("QI21_API_KEY", "")

# Hard ceiling on a single request. Cell 5 measures the real cost; this is
# only a backstop so a runaway 4K render cannot pin the card for an hour.
MAX_STEPS = int(os.environ.get("QI21_MAX_STEPS", "50"))
MAX_PIXELS = int(os.environ.get("QI21_MAX_PIXELS", str(2048 * 2048)))

# THE lock. See design note 1 — do not remove this.
GPU_LOCK = asyncio.Lock()

app = FastAPI(title="Qwen-Image-2.1", version="1.0")
PIPE = None
BOOT_T = time.time()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="what to draw")
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=MAX_STEPS)
    cfg: float = Field(default=1.0, ge=1.0, le=8.0,
                       description="true_cfg_scale; 1.0 is the model's documented "
                                   "default and is IGNORED-style guidance-free sampling. "
                                   "Above 1.0 makes the DiT run twice per step (2x cost).")
    seed: int = Field(default=-1, description="-1 = random")
    negative_prompt: str = Field(default="",
                                 description="only used when cfg > 1 — the pipeline "
                                             "ignores it at true_cfg_scale=1")


async def require_token(authorization: str = Header(default="")) -> None:
    """Bearer-token gate. 401 with WWW-Authenticate, per the HTTP spec."""
    if not TOKEN:
        raise HTTPException(500, "QI21_API_KEY is not set on the server")
    expected = f"Bearer {TOKEN}"
    # constant-time compare: this key is a shared secret and the endpoint is public
    if len(authorization) != len(expected) or not _consteq(authorization, expected):
        raise HTTPException(401, "bad or missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


def _consteq(a: str, b: str) -> bool:
    acc = 0
    for x, y in zip(a.encode(), b.encode()):
        acc |= x ^ y
    return acc == 0


@app.get("/health")
async def health():
    return {
        "status": "ok" if PIPE is not None else "loading",
        "model": "Qwen-Image-2.1",
        "dtype": "bfloat16",
        "resident_gib": round(torch.cuda.memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None,
        "uptime_s": round(time.time() - BOOT_T, 1),
        "busy": GPU_LOCK.locked(),
    }


@app.post("/generate", dependencies=[Depends(require_token)])
@app.post("/v1/images/generations", dependencies=[Depends(require_token)])
async def generate(req: GenerateRequest):
    if PIPE is None:
        raise HTTPException(503, "model still loading")

    if req.width * req.height > MAX_PIXELS:
        raise HTTPException(400, f"too many pixels: cap is {MAX_PIXELS}")

    seed = req.seed if req.seed >= 0 else int(torch.Generator("cuda").initial_seed())
    gen = torch.Generator("cuda").manual_seed(seed)

    # Wait politely, but do not queue without bound — see design note 2.
    try:
        await asyncio.wait_for(GPU_LOCK.acquire(), timeout=float(os.environ.get("QI21_QUEUE_S", "5")))
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=429,
            content={"error": "busy — another generation is running",
                     "retry_after_s": int(os.environ.get("QI21_QUEUE_S", "5"))},
            headers={"Retry-After": os.environ.get("QI21_QUEUE_S", "5")},
        )

    try:
        t0 = time.time()
        kwargs = dict(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            true_cfg_scale=req.cfg,
            generator=gen,
            # Pinned explicitly, not left to the default. The docs warn that
            # toggling this "does not reproduce the same image bit-for-bit in
            # reduced precision" — a 1-ULP difference at block 1 gets amplified
            # through 32 blocks and every step. Pinning it means a given
            # (prompt, seed, steps) gives a reproducible image instead of one
            # that drifts with library defaults.
            use_kv_cache=True,
        )
        # Two traps, both from the diffusers docs for this pipeline:
        #  - the kwarg is `true_cfg_scale`; there is NO `guidance_scale` on
        #    QwenImage21Pipeline, passing one raises TypeError.
        #  - "negative_prompt ... Ignored when true_cfg_scale is not greater
        #    than 1." Since the default is 1.0, forwarding a negative prompt
        #    at the default is a silent no-op. Only send it when cfg > 1.
        if req.negative_prompt and req.cfg > 1.0:
            kwargs["negative_prompt"] = req.negative_prompt
        # The blocking CUDA call has to leave the event loop alone, or the
        # health endpoint stops answering while a render is in flight.
        image = await asyncio.to_thread(lambda: PIPE(**kwargs).images[0])
        dt = time.time() - t0

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        GPU_LOCK.release()

    # Response shape is the OpenAI /v1/images/generations one — DALL·E-shaped:
    #   {"created": <epoch>, "data": [{"b64_json": ..., "url": null, ...}]}
    # vLLM-Omni serves Qwen-Image-2.1 at exactly this shape and its docs drive
    # it through the OpenAI Python SDK (client.images.generate), so matching it
    # means a real SDK client works against this server unmodified instead of
    # needing a shim. The extra keys after "data" are additions, not
    # replacements — an OpenAI client ignores unknown fields, and they are what
    # cell 6c and you read for timing. (Note: vLLM-Omni's own Qwen-Image-2.1
    # support is unmerged and was verified only on GB300 — we match the
    # CONVENTION, we are not standing on that implementation.)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": b64, "url": None, "revised_prompt": None}],
        # --- non-OpenAI extras, additive ---
        "seed": seed,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "seconds": round(dt, 2),
        "seconds_per_step": round(dt / req.steps, 3),
    }


def load() -> None:
    """Load the pipeline. Called before uvicorn starts so /health is honest."""
    global PIPE
    from diffusers import QwenImage21Pipeline

    print(f"loading {WORK} -> cuda (bfloat16, no offload)", flush=True)
    t0 = time.time()
    PIPE = QwenImage21Pipeline.from_pretrained(str(WORK), torch_dtype=torch.bfloat16).to("cuda")
    gib = torch.cuda.memory_allocated() / 2**30
    print(f"MODEL_READY in {time.time()-t0:.1f}s — {gib:.1f} GiB resident", flush=True)


if __name__ == "__main__":
    import uvicorn

    if not torch.cuda.is_available():
        raise SystemExit("!! no CUDA — this server needs the GPU Studio")
    if not TOKEN:
        raise SystemExit("!! QI21_API_KEY is not set — refusing to serve unauthenticated")
    load()
    # ONE worker on purpose. The lock above is per-process; more workers would
    # each load their own 33.1 GB copy and defeat both the lock and the VRAM.
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1, log_level="info")
QISERVER_EOF
  echo "wrote qi21_server.py from the embedded copy"
fi

# ---- 1. credentials -------------------------------------------------------
# The key lives in the environment and is printed once. It is NOT written into
# qi21_server.py and NOT committed. If you re-run this cell you get a NEW key
# and the old one stops working — that is intentional.
if [ -z "$QI21_API_KEY" ]; then
  QI21_API_KEY="qi21-$(head -c 18 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  export QI21_API_KEY
fi

# ---- 2. deadline watchdog -------------------------------------------------
# Kills the server AND the tunnel at the deadline. This is the billing guard.
cat > "$WATCH" <<EOF
#!/bin/sh
LIMIT=\$(( ${MAX_MIN} * 60 )); T0=\$(date +%s)
while true; do
  NOW=\$(date +%s)
  if [ \$((NOW - T0)) -ge \$LIMIT ]; then
    pkill -f qi21_server.py
    pkill -f "ngrok http"
    echo "\$(date) WATCHDOG: ${MAX_MIN} min reached — server + tunnel stopped" >> "$LOG"
    exit 0
  fi
  sleep 60
done
EOF
chmod +x "$WATCH"

# Kills and launches are SEPARATE commands on purpose. A combined
# `pkill -f X && nohup X &` can match its own launch text and kill the new
# process (hard-won rule, see HANDOFF.md).
pkill -f qi21_watchdog_marker 2>/dev/null
pkill -f qi21_server.py 2>/dev/null
pkill -f "ngrok http" 2>/dev/null
sleep 3

# ---- 3. launch ------------------------------------------------------------
QI21_API_KEY="$QI21_API_KEY" QI21_WORK="$WORK" \
  nohup python3 "$WORK/qi21_server.py" > "$LOG" 2>&1 &
echo "server launching — log $LOG"

# The watchdog runs as a separate detached process and matches on the log path,
# which is unique to this session (never on a pattern the launch line shares).
nohup sh -c "exec -a qi21_watchdog_marker $WATCH" > /dev/null 2>&1 &
echo "watchdog armed — hard stop at ${MAX_MIN} min"

sleep 20
echo ""
echo "=== log ==="
tail -6 "$LOG" 2>/dev/null

echo ""
echo "=== is it up? (no token needed for /health) ==="
curl -s --max-time 10 http://127.0.0.1:8080/health || echo "(not answering yet — the 33.1 GB load takes a minute; re-run this cell's tail)"

echo ""
echo "############################################################"
echo "#  SERVER UP (local only — still no public URL)"
echo "#  API KEY : $QI21_API_KEY"
echo "#  local   : http://127.0.0.1:8080"
echo ""
echo "#  next: cell_06b_tunnel.sh to expose it publicly"
echo "#  kill  : pkill -f qi21_server.py   (stops billing immediately)"
echo "############################################################"
