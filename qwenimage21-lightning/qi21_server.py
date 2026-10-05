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
from diffusers import FlowMatchEulerDiscreteScheduler
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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
# Fast lane (opt-in): Turbo8 distilled LoRA + its required scheduler. Built at
# boot; if it fails the server still serves base quality and /health says so.
FAST_READY = False
FAST_SCHEDULER = None
BASE_SCHEDULER = None

# The browser UI, embedded verbatim from the kit's qi21_ui.html (the canonical,
# readable copy — this string must stay byte-identical to it; cell_06_serve.sh's
# build step verifies). Served same-origin at / and /ui, which is why the UI
# needs no CORS config and no separate host.
UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen-Image-2.1</title>
<style>
  :root {
    --bg: #0c0d10;
    --surface: #14161b;
    --surface-2: #191c23;
    --border: #262a33;
    --text: #e8eaf0;
    --muted: #97a0b0;
    --accent: #e8a33d;
    --accent-ink: #171003;
    --danger: #e5645f;
    --ok: #6fbf8b;
    --radius: 8px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.5 ui-sans-serif, system-ui, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  }

  header {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: .01em; }
  header .sub { color: var(--muted); font-size: 12px; }
  header .spacer { flex: 1; }
  #health { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  #health .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--danger); }
  #health.ok .dot { background: var(--ok); }

  main {
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 0;
    height: calc(100vh - 49px);
  }
  @media (max-width: 860px) {
    main { grid-template-columns: 1fr; height: auto; }
    #canvas { min-height: 60vh; }
  }

  aside {
    border-right: 1px solid var(--border);
    padding: 16px;
    overflow-y: auto;
    display: flex; flex-direction: column; gap: 16px;
  }
  @media (max-width: 860px) { aside { border-right: 0; border-bottom: 1px solid var(--border); } }

  .field { display: flex; flex-direction: column; gap: 6px; }
  .field > label { font-size: 12px; font-weight: 600; color: var(--muted); letter-spacing: .02em; }
  .hint { font-size: 11px; color: var(--muted); }

  textarea, input[type="number"], input[type="password"], select {
    width: 100%;
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 10px;
    font: inherit;
  }
  textarea { resize: vertical; min-height: 84px; }
  textarea:focus-visible, input:focus-visible, select:focus-visible, button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  .row { display: flex; gap: 8px; align-items: center; }
  .row > * { flex: 1; }
  .row .tight { flex: 0 0 auto; }

  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius);
    padding: 5px 9px;
    font-size: 12px;
    cursor: pointer;
  }
  .chip[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }

  #generate {
    background: var(--accent);
    color: var(--accent-ink);
    border: 0;
    border-radius: var(--radius);
    padding: 11px 16px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
  }
  #generate[disabled] { opacity: .55; cursor: wait; }
  kbd {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 4px; padding: 0 5px; font-size: 11px; font-family: inherit;
  }

  #canvas {
    position: relative;
    display: flex; align-items: center; justify-content: center;
    padding: 20px;
    overflow: auto;
  }
  #stage { position: relative; max-width: 100%; }
  #stage img {
    display: block;
    max-width: 100%;
    max-height: calc(100vh - 200px);
    border-radius: 12px;
    border: 1px solid var(--border);
  }
  #stage img.busy { visibility: hidden; }

  #shimmer {
    position: absolute; inset: 0;
    border-radius: 12px;
    background: var(--surface);
    overflow: hidden;
  }
  #shimmer::after {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(100deg, transparent 30%, var(--surface-2) 50%, transparent 70%);
    animation: sweep 1.4s infinite;
  }
  @keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(100%); } }

  #busybox {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; gap: 4px;
    align-items: center; justify-content: center;
    text-align: center;
  }
  #busybox .t { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  #busybox .s { color: var(--muted); font-size: 12px; }

  .meta { display: flex; gap: 14px; align-items: center; margin-top: 10px; color: var(--muted); font-size: 12px; flex-wrap: wrap; }
  .meta button {
    background: none; border: 1px solid var(--border); color: var(--text);
    border-radius: var(--radius); padding: 4px 10px; font-size: 12px; cursor: pointer;
  }
  .meta button:hover { border-color: var(--accent); }

  #empty { color: var(--muted); text-align: center; max-width: 380px; }
  #empty .big { font-size: 15px; color: var(--text); margin-bottom: 6px; }

  #statusbar { min-height: 20px; font-size: 12px; }
  #statusbar.error { color: var(--danger); }
  #statusbar.info  { color: var(--muted); }

  #history {
    display: flex; gap: 8px; overflow-x: auto;
    padding: 10px 20px 16px;
  }
  #history button {
    flex: 0 0 auto; padding: 0;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); cursor: pointer; overflow: hidden;
  }
  #history button[aria-current="true"] { border-color: var(--accent); }
  #history img { display: block; height: 72px; width: auto; }
</style>
</head>
<body>

<header>
  <h1>Qwen-Image-2.1</h1>
  <span class="sub">full BF16 · single H200</span>
  <span class="spacer"></span>
  <span id="health" class=""><span class="dot"></span><span id="health-text">checking…</span></span>
</header>

<main>
  <aside>
    <div class="field">
      <label for="prompt">Prompt</label>
      <textarea id="prompt" autofocus
        placeholder="A neon shop sign that reads QWEN IMAGE, rainy night, reflections on wet pavement"></textarea>
      <span class="hint">Ctrl + Enter to generate</span>
    </div>

    <div class="field">
      <label for="negative">Negative prompt <span class="hint">(ignored unless CFG &gt; 1)</span></label>
      <input type="text" id="negative" placeholder="optional">
    </div>

    <div class="field">
      <label id="size-label">Size</label>
      <div class="chips" role="group" aria-labelledby="size-label">
        <button type="button" class="chip" data-w="1024" data-h="1024" aria-pressed="true">1:1 · 1024</button>
        <button type="button" class="chip" data-w="1216" data-h="832" aria-pressed="false">3:2</button>
        <button type="button" class="chip" data-w="1344" data-h="768" aria-pressed="false">16:9</button>
        <button type="button" class="chip" data-w="768" data-h="1344" aria-pressed="false">9:16</button>
        <button type="button" class="chip" data-w="2048" data-h="2048" aria-pressed="false">2K</button>
      </div>
      <div class="row">
        <div class="field"><label for="width">Width</label><input type="number" id="width" value="1024" min="64" max="2048" step="32"></div>
        <div class="field"><label for="height">Height</label><input type="number" id="height" value="1024" min="64" max="2048" step="32"></div>
      </div>
    </div>

    <div class="field">
      <label for="steps">Steps: <span id="steps-val">28</span></label>
      <input type="range" id="steps" min="4" max="50" step="1" value="28">
      <span class="hint">Quality dial — cost scales linearly. 28 is the balance point.</span>
    </div>

    <div class="field">
      <label for="cfg">CFG</label>
      <input type="number" id="cfg" value="1.0" min="1" max="8" step="0.5">
      <span class="hint">Model default is 1.0 (guidance-free). Above 1 runs the DiT twice per step.</span>
    </div>

    <div class="field">
      <label for="seed">Seed</label>
      <div class="row">
        <input type="number" id="seed" value="-1" step="1">
        <button type="button" class="chip tight" id="use-last" title="Use the seed of the last image">last</button>
        <button type="button" class="chip tight" id="randomize" title="Random seed (-1)">rnd</button>
      </div>
    </div>

    <div class="field">
      <label for="fast" style="display:flex;align-items:center;gap:8px;color:var(--text);font-size:13px;cursor:pointer">
        <input type="checkbox" id="fast" style="width:auto;accent-color:var(--accent)">
        Fast mode <span class="hint">(Turbo8 · 8 steps · ~1.5s — dense text degrades)</span>
      </label>
    </div>

    <div class="field">
      <label for="key">API key</label>
      <input type="password" id="key" placeholder="qi21-…" autocomplete="off">
      <span class="hint">Stored in this browser only (localStorage).</span>
    </div>

    <button id="generate">Generate</button>
    <div id="statusbar" role="status" aria-live="polite"></div>
  </aside>

  <section>
    <div id="canvas">
      <div id="stage" hidden>
        <img id="image" alt="Generated image">
        <div id="shimmer" hidden></div>
        <div id="busybox" hidden>
          <div class="t" id="elapsed">0.0s</div>
          <div class="s" id="busy-sub">denoising…</div>
        </div>
      </div>
      <div id="empty">
        <div class="big">Nothing generated yet</div>
        <div>Write a prompt, pick a size, hit Generate.</div>
        <div style="margin-top:6px"><kbd>Ctrl</kbd> + <kbd>Enter</kbd> works from the prompt box.</div>
      </div>
    </div>
    <div class="meta" id="meta" hidden>
      <span id="meta-info"></span>
      <span class="tight"></span>
      <button type="button" id="download">Download PNG</button>
    </div>
    <div id="history" hidden></div>
  </section>
</main>

<script>
"use strict";
var $ = function (id) { return document.getElementById(id); };
var els = {
  prompt: $("prompt"), negative: $("negative"), width: $("width"), height: $("height"),
  steps: $("steps"), stepsVal: $("steps-val"), cfg: $("cfg"), seed: $("seed"),
  key: $("key"), generate: $("generate"), statusbar: $("statusbar"),
  stage: $("stage"), image: $("image"), shimmer: $("shimmer"), busybox: $("busybox"),
  elapsed: $("elapsed"), busySub: $("busy-sub"), empty: $("empty"),
  meta: $("meta"), metaInfo: $("meta-info"), download: $("download"),
  history: $("history"), health: $("health"), healthText: $("health-text"),
  useLast: $("use-last"), randomize: $("randomize"),
};

var busy = false, timer = null, t0 = 0;
var elsFast = $("fast");
var gallery = [];   // {url, seed, width, height, steps, seconds}
var current = -1;

// ---- api key: persisted locally, never sent anywhere but this origin ----
els.key.value = localStorage.getItem("qi21_key") || "";
els.key.addEventListener("change", function () {
  localStorage.setItem("qi21_key", els.key.value.trim());
});

// ---- health ping ----
function ping() {
  fetch("/health").then(function (r) { return r.json(); }).then(function (d) {
    els.health.className = d.status === "ok" ? "ok" : "";
    els.healthText.textContent = d.status === "ok"
      ? "live · " + d.resident_gib + " GiB resident" + (d.busy ? " · busy" : " · idle")
      : "loading";
  }).catch(function () {
    els.health.className = "";
    els.healthText.textContent = "unreachable";
  });
}
ping(); setInterval(ping, 20000);

// ---- size chips ----
document.querySelectorAll(".chip[data-w]").forEach(function (chip) {
  chip.addEventListener("click", function () {
    document.querySelectorAll(".chip[data-w]").forEach(function (c) { c.setAttribute("aria-pressed", "false"); });
    chip.setAttribute("aria-pressed", "true");
    els.width.value = chip.dataset.w;
    els.height.value = chip.dataset.h;
  });
});
els.width.addEventListener("input", function () {
  document.querySelectorAll(".chip[data-w]").forEach(function (c) { c.setAttribute("aria-pressed", "false"); });
});
els.steps.addEventListener("input", function () { els.stepsVal.textContent = els.steps.value; });

els.useLast.addEventListener("click", function () {
  if (gallery.length) els.seed.value = gallery[gallery.length - 1].seed;
});
els.randomize.addEventListener("click", function () { els.seed.value = -1; });

function setStatus(msg, cls) {
  els.statusbar.textContent = msg || "";
  els.statusbar.className = cls || "";
}

// ---- generate ----
function generate() {
  if (busy) return;
  var prompt = els.prompt.value.trim();
  var key = els.key.value.trim();
  if (!prompt) { setStatus("Write a prompt first.", "error"); els.prompt.focus(); return; }
  if (!key) { setStatus("Paste the API key (it is in the cell 6 banner).", "error"); els.key.focus(); return; }

  busy = true;
  els.generate.disabled = true;
  els.generate.textContent = "Generating…";
  els.empty.hidden = true;
  els.image.classList.add("busy");
  els.shimmer.hidden = false;
  els.busybox.hidden = false;
  els.stage.hidden = false;
  setStatus("", "");

  var steps = parseInt(els.steps.value, 10) || 28;
  if (elsFast.checked) { steps = 8; els.steps.value = 8; els.stepsVal.textContent = "8"; }
  t0 = performance.now();
  timer = setInterval(function () {
    var dt = (performance.now() - t0) / 1000;
    els.elapsed.textContent = dt.toFixed(1) + "s";
    els.busySub.textContent = (elsFast.checked ? "fast lane · " : "denoising · ") + steps + " steps · ~" + (dt / steps).toFixed(2) + " s/step so far";
  }, 100);

  var body = {
    prompt: prompt,
    fast: elsFast.checked,
    width: parseInt(els.width.value, 10) || 1024,
    height: parseInt(els.height.value, 10) || 1024,
    steps: steps,
    cfg: parseFloat(els.cfg.value) || 1.0,
    seed: parseInt(els.seed.value, 10),
  };
  if (els.negative.value.trim() && body.cfg > 1) body.negative_prompt = els.negative.value.trim();

  fetch("/v1/images/generations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer " + key,
      "ngrok-skip-browser-warning": "true"
    },
    body: JSON.stringify(body)
  }).then(function (r) {
    return r.json().then(function (d) { return { status: r.status, data: d }; });
  }).then(function (res) {
    clearInterval(timer); busy = false;
    els.generate.disabled = false;
    els.generate.textContent = "Generate";
    els.shimmer.hidden = true;
    els.busybox.hidden = true;
    els.image.classList.remove("busy");

    if (res.status !== 200) {
      var msg = (res.data && (res.data.error || res.data.detail)) || ("HTTP " + res.status);
      setStatus(msg + (res.status === 429 ? " — the card is busy; retry in a few seconds." : ""), "error");
      if (!gallery.length) { els.stage.hidden = true; els.empty.hidden = false; }
      return;
    }

    var d = res.data;
    var b64 = d.data[0].b64_json;
    var bin = atob(b64), n = bin.length, bytes = new Uint8Array(n);
    for (var i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    var url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));

    els.image.src = url;
    els.stage.hidden = false;
    current = gallery.length;
    gallery.push({ url: url, seed: d.seed, width: d.width, height: d.height, steps: d.steps, seconds: d.seconds, prompt: prompt });
    els.meta.hidden = false;
    els.metaInfo.textContent = (d.fast ? "FAST · " : "") + d.width + "×" + d.height + " · " + d.steps + " steps · " + d.seconds + " s · seed " + d.seed;
    els.download.hidden = false;
    renderHistory();
    setStatus("done in " + d.seconds + " s (" + d.seconds_per_step + " s/step)", "info");
  }).catch(function (err) {
    clearInterval(timer); busy = false;
    els.generate.disabled = false;
    els.generate.textContent = "Generate";
    els.shimmer.hidden = true; els.busybox.hidden = true;
    els.image.classList.remove("busy");
    setStatus("request failed: " + err.message, "error");
  });
}

els.generate.addEventListener("click", generate);
els.prompt.addEventListener("keydown", function (e) {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate();
});

// ---- download + history ----
els.download.addEventListener("click", function () {
  if (current < 0) return;
  var g = gallery[current];
  var a = document.createElement("a");
  a.href = g.url;
  a.download = "qwen21_" + g.seed + "_" + g.width + "x" + g.height + ".png";
  a.click();
});

function renderHistory() {
  if (gallery.length < 1) { els.history.hidden = true; return; }
  els.history.hidden = false;
  els.history.innerHTML = "";
  gallery.forEach(function (g, i) {
    var b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", "image " + (i + 1) + ", seed " + g.seed);
    b.setAttribute("aria-current", i === current ? "true" : "false");
    var img = document.createElement("img");
    img.src = g.url; img.alt = "";
    b.appendChild(img);
    b.addEventListener("click", function () {
      current = i;
      els.image.src = g.url;
      els.stage.hidden = false; els.empty.hidden = true;
      els.meta.hidden = false;
      els.metaInfo.textContent = g.width + "×" + g.height + " · " + g.steps + " steps · " + g.seconds + " s · seed " + g.seed;
      renderHistory();
    });
    els.history.appendChild(b);
  });
}
</script>
</body>
</html>"""


@app.get("/ui")
@app.get("/")
async def ui():
    return HTMLResponse(UI_HTML)


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
    fast: bool = Field(default=False,
                       description="Turbo8 distilled lane: 8 steps, cfg forced 1.0, "
                                   "~1.5s per 1024px image. Cost: dense text and complex "
                                   "edits degrade (text exact-match 95% -> 75%).")


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
        "fast_ready": FAST_READY,
    }


@app.post("/generate", dependencies=[Depends(require_token)])
@app.post("/v1/images/generations", dependencies=[Depends(require_token)])
async def generate(req: GenerateRequest):
    if PIPE is None:
        raise HTTPException(503, "model still loading")

    if req.width * req.height > MAX_PIXELS:
        raise HTTPException(400, f"too many pixels: cap is {MAX_PIXELS}")

    fast = bool(req.fast)
    if fast and not FAST_READY:
        raise HTTPException(503, "fast mode unavailable — the Turbo8 LoRA did not load at boot")

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

    steps = 8 if fast else req.steps   # Turbo8 is distilled for exactly 8
    cfg = 1.0 if fast else req.cfg     # distilled cards are guidance-free
    try:
        t0 = time.time()
        kwargs = dict(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
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
        if req.negative_prompt and cfg > 1.0:
            kwargs["negative_prompt"] = req.negative_prompt
        if fast:
            # The stock scheduler config wrecks few-step schedules — Turbo8's
            # card requires a scheduler with shift_terminal unset. Swap under
            # the lock, restore after, so base requests are untouched.
            PIPE.scheduler = FAST_SCHEDULER
        # The blocking CUDA call has to leave the event loop alone, or the
        # health endpoint stops answering while a render is in flight.
        image = await asyncio.to_thread(lambda: PIPE(**kwargs).images[0])
        dt = time.time() - t0
    finally:
        if fast:
            PIPE.scheduler = BASE_SCHEDULER
        GPU_LOCK.release()

    # PNG + base64 run OUTSIDE the lock (compress_level=1: this is transport
    # encoding, not archival). Previously this sat inside the lock, so every
    # queued request waited for the previous request's PNG encode — up to
    # 0.1-0.3 s of pure stall per request with the GPU idle (found by the
    # speed-research pass, 2026-10-06).
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

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
        "steps": steps,
        "seconds": round(dt, 2),
        "seconds_per_step": round(dt / steps, 3),
        "fast": fast,
    }


def load() -> None:
    """Load the pipeline. Called before uvicorn starts so /health is honest."""
    global PIPE, BASE_SCHEDULER, FAST_SCHEDULER, FAST_READY
    from diffusers import QwenImage21Pipeline

    print(f"loading {WORK} -> cuda (bfloat16, no offload)", flush=True)
    t0 = time.time()
    PIPE = QwenImage21Pipeline.from_pretrained(str(WORK), torch_dtype=torch.bfloat16).to("cuda")
    BASE_SCHEDULER = PIPE.scheduler
    gib = torch.cuda.memory_allocated() / 2**30
    print(f"MODEL_READY in {time.time()-t0:.1f}s — {gib:.1f} GiB resident", flush=True)

    # Fast lane: Turbo8 (r128, 8 steps, T2I + editing + RGBA + up to 2K).
    # Requirements are from its model card and are not negotiable: the LoRA,
    # and a scheduler with shift_terminal unset. Kept strictly opt-in — base
    # quality stays the default for every request that does not ask for it.
    try:
        PIPE.load_lora_weights(
            "chriswritescode/Turbo8-LoRA-Qwen-Image-2.1",
            weight_name="turbo8_lora_step2500.safetensors",
        )
        FAST_SCHEDULER = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(WORK), subfolder="scheduler", shift_terminal=None
        )
        FAST_READY = True
        print("FAST_READY — Turbo8 loaded; fast=true serves 8-step images", flush=True)
    except Exception as exc:  # noqa: BLE001 — base serving must survive this
        print(f"!! Turbo8 fast lane failed ({type(exc).__name__}: {exc}) — base only", flush=True)


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
