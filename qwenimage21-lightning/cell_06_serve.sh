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
WORKERS="${QI21_WORKERS:-3}"
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
# ALWAYS overwrite: an earlier "keep if present" guard silently relaunched the
# OLD server after an update — the file existed, so the new embedded bytes
# never landed and /ui 404'd (seen live 2026-10-05). The kit copy is the only
# source of truth; do not hand-edit the $WORK copy.
cat > qi21_server.py <<'QISERVER_EOF'
#!/usr/bin/env python3
"""Qwen-Image-2.1 HTTP server — full BF16, N workers, never-failing job queue.

Written by cell_06_serve.sh (the canonical readable copy lives beside it in the
kit; the cell embeds these exact bytes).

ARCHITECTURE — what changed and why (2026-10-06):

1. **The queue is the scheduler now.** POST /generate never renders inline and
   NEVER rejects for capacity: it inserts a row into a SQLite (WAL) job store
   and returns 202 {job_id, queue_position}. No 429s exist anywhere. This is
   the fal.ai queue pattern (their docs: "requests in the queue are never
   dropped", "no queue size limit").

2. **SQLite in WAL mode is the cross-worker queue.** With uvicorn workers=N
   there are N separate processes; asyncio.Queue/Lock are per-process and
   cannot route jobs between them (documented FastAPI pitfall). One DB file on
   disk, opened by all workers: BEGIN IMMEDIATE gives a serialized atomic
   claim, ORDER BY seq gives strict FIFO fairness — the closest queue number
   takes the next free worker, exactly the required behavior. The DB survives
   worker crashes and Studio restarts; pending jobs are just rows (~300 bytes).

3. **N workers, one pipeline each.** QI21_WORKERS (default 3) uvicorn workers,
   each loading its own 30.2 GiB pipeline in its startup hook (3 x 30.2 = 90.6
   GiB resident; worst measured mix ~130 GiB < 139.8). Each runs a dispatcher
   task: claim oldest queued row -> render -> write result PNG -> mark done.
   The in-process GPU_LOCK stays as defense-in-depth (pipelines are not
   thread-safe), but HTTP handlers never touch it — only dispatchers do.

4. **PNG encoding never blocks anything.** Encoded by the dispatcher after the
   render, compress_level=1 (transport, not archival). Results live as PNG
   files under WORK/results/{job_id}.png with a 24h TTL cleanup (Replicate
   deletes after 1h; we are kinder).

5. **OpenAI compatibility kept:** /v1/images/generations submits a job, then
   polls the store for up to QI21_SYNC_WAIT_S (default 120 s) and returns the
   OpenAI-shaped body (data[0].b64_json). If the queue is deeper than the
   wait, it returns the 202 job envelope instead of an error — the queue can
   delay a response, never reject it.

6. **Fast lane unchanged and opt-in:** fast=true forces Turbo8 8 steps, cfg
   1.0, its shift_terminal-unset scheduler (swapped under the lock, restored
   after). Base quality remains the default for every other request.

Fairness/positions: position = COUNT(queued rows with smaller seq). Recomputed
on every poll — always fresh, never drifts. Human numbering adds one.
"""

import asyncio
import base64
import io
import json
import os
import pathlib
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

WORK = pathlib.Path(os.environ.get("QI21_WORK", pathlib.Path.home() / "qwenimage21"))
PORT = int(os.environ.get("QI21_PORT", "8080"))
TOKEN = os.environ.get("QI21_API_KEY", "")
WORKERS = int(os.environ.get("QI21_WORKERS", "3"))

# Hard ceilings on a single request. Cell 5 measures the real cost; this is
# only a backstop so a runaway 4K render cannot pin the card for an hour.
MAX_STEPS = int(os.environ.get("QI21_MAX_STEPS", "50"))
MAX_PIXELS = int(os.environ.get("QI21_MAX_PIXELS", str(2048 * 2048)))
SYNC_WAIT_S = float(os.environ.get("QI21_SYNC_WAIT_S", "120"))
RENDER_LEASE_S = float(os.environ.get("QI21_RENDER_LEASE_S", "900"))
RESULT_TTL_S = float(os.environ.get("QI21_RESULT_TTL_S", "86400"))
MAX_REQUEUES = 6

RESULTS = WORK / "results"
# Measured transient VRAM above resident, GiB per megapixel of output
# (1024²: +6.6, 2048²: +26.3 — from cell 5's bench). Used by the dispatcher to
# claim a job only when the card actually has room for it, so big renders WAIT
# for calm VRAM instead of OOM-churning through retries.
TRANSIENT_GIB_PER_MP = 6.6
VRAM_SAFETY_GIB = 2.0
DB_PATH = WORK / "jobs.db"

# THE lock: pipelines are not thread-safe (scheduler state mutates per call).
# Only dispatcher coroutines touch it; with one dispatcher per worker process
# it is trivially uncontended — kept so that ever running >1 dispatcher in a
# process stays safe.
GPU_LOCK = asyncio.Lock()

PIPE = None
BASE_SCHEDULER = None
FAST_SCHEDULER = None
FAST_READY = False
BOOT_T = time.time()
WORKER_ID = f"worker-{os.getpid()}"

# The browser UI, embedded verbatim from the kit's qi21_ui.html (byte-identity
# is verified by cell_06_serve.sh's build step). Same-origin => no CORS.
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
  #busybox .q { color: var(--accent); font-size: 15px; font-weight: 700; margin-bottom: 2px; }

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
  <span class="sub">full BF16 · queued · never rejects</span>
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
          <div class="q" id="queue-pos" hidden></div>
          <div class="t" id="elapsed">0.0s</div>
          <div class="s" id="busy-sub"></div>
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
  key: $("key"), fast: $("fast"), generate: $("generate"), statusbar: $("statusbar"),
  stage: $("stage"), image: $("image"), shimmer: $("shimmer"), busybox: $("busybox"),
  elapsed: $("elapsed"), busySub: $("busy-sub"), queuePos: $("queue-pos"), empty: $("empty"),
  meta: $("meta"), metaInfo: $("meta-info"), download: $("download"),
  history: $("history"), health: $("health"), healthText: $("health-text"),
  useLast: $("use-last"), randomize: $("randomize"),
};

var busy = false, timer = null, t0 = 0, pollTimer = null;
var gallery = [];   // {url, seed, width, height, steps, seconds, fast}
var current = -1;

// ---- api key: persisted locally, never sent anywhere but this origin ----
els.key.value = localStorage.getItem("qi21_key") || "";
els.key.addEventListener("change", function () {
  localStorage.setItem("qi21_key", els.key.value.trim());
});
function authHeaders(extra) {
  var h = extra || {};
  h["Authorization"] = "Bearer " + els.key.value.trim();
  h["ngrok-skip-browser-warning"] = "true";
  return h;
}

// ---- health ping ----
function ping() {
  fetch("/health").then(function (r) { return r.json(); }).then(function (d) {
    els.health.className = d.status === "ok" ? "ok" : "";
    var q = (typeof d.queued === "number" && d.queued >= 0) ? " · queue " + d.queued : "";
    els.healthText.textContent = d.status === "ok"
      ? "live · " + d.resident_gib + " GiB · " + d.workers + " workers" + q
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

function stopBusy() {
  busy = false;
  if (timer) { clearInterval(timer); timer = null; }
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  els.generate.disabled = false;
  els.generate.textContent = "Generate";
  els.shimmer.hidden = true;
  els.busybox.hidden = true;
  els.image.classList.remove("busy");
}

// ---- submit + poll (never-fail queue: 202 -> job id -> status) ----
function generate() {
  if (busy) return;
  var prompt = els.prompt.value.trim();
  var key = els.key.value.trim();
  if (!prompt) { setStatus("Write a prompt first.", "error"); els.prompt.focus(); return; }
  if (!key) { setStatus("Paste the API key (it is in the cell 6 banner).", "error"); els.key.focus(); return; }

  busy = true;
  els.generate.disabled = true;
  els.generate.textContent = "Queued…";
  els.empty.hidden = true;
  els.image.classList.add("busy");
  els.shimmer.hidden = false;
  els.busybox.hidden = false;
  els.stage.hidden = false;
  els.queuePos.hidden = false;
  els.queuePos.textContent = "submitting…";
  els.busySub.textContent = "";
  setStatus("", "");
  t0 = performance.now();
  timer = setInterval(function () {
    els.elapsed.textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s";
  }, 100);

  var body = {
    prompt: prompt,
    width: parseInt(els.width.value, 10) || 1024,
    height: parseInt(els.height.value, 10) || 1024,
    steps: parseInt(els.steps.value, 10) || 28,
    cfg: parseFloat(els.cfg.value) || 1.0,
    seed: parseInt(els.seed.value, 10),
    fast: els.fast.checked,
  };
  if (body.fast) { body.steps = 8; els.steps.value = 8; els.stepsVal.textContent = "8"; }
  if (els.negative.value.trim() && body.cfg > 1) body.negative_prompt = els.negative.value.trim();

  fetch("/generate", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body)
  }).then(function (r) {
    return r.json().then(function (d) { return { status: r.status, data: d }; });
  }).then(function (res) {
    if (res.status === 401) {
      stopBusy();
      if (!gallery.length) { els.stage.hidden = true; els.empty.hidden = false; }
      setStatus("bad API key — paste the one from the cell 6 banner.", "error");
      return;
    }
    if (res.status !== 202) {
      stopBusy();
      if (!gallery.length) { els.stage.hidden = true; els.empty.hidden = false; }
      setStatus((res.data && (res.data.error || res.data.detail)) || ("HTTP " + res.status), "error");
      return;
    }
    setStatus("accepted — you are in the queue", "info");
    poll(res.data.job_id, 600);
  }).catch(function (err) {
    stopBusy();
    if (!gallery.length) { els.stage.hidden = true; els.empty.hidden = false; }
    setStatus("request failed: " + err.message, "error");
  });
}

function poll(jobId, waitMs) {
  pollTimer = setTimeout(function () {
    fetch("/jobs/" + jobId, { headers: authHeaders() })
      .then(function (r) { return r.json().then(function (d) { return { status: r.status, data: d }; }); })
      .then(function (res) {
        if (res.status !== 200) {
          if (res.status === 404) setStatus("job expired — generate again.", "error");
          else setStatus("status check failed: HTTP " + res.status, "error");
          stopBusy();
          return;
        }
        var d = res.data;
        if (d.status === "queued") {
          var n = (typeof d.queue_position === "number") ? d.queue_position + 1 : "?";
          els.queuePos.hidden = false;
          els.queuePos.textContent = "you are #" + n + " in queue";
          els.busySub.textContent = "your slot is reserved — queued jobs are never dropped";
          els.generate.textContent = "Queued…";
          poll(jobId, n <= 3 ? 800 : 2000);
          return;
        }
        if (d.status === "rendering") {
          els.queuePos.hidden = true;
          els.busySub.textContent = "rendering on a free worker…";
          els.generate.textContent = "Rendering…";
          poll(jobId, 900);
          return;
        }
        if (d.status === "done") {
          finishDone(d);
          return;
        }
        // error | canceled
        stopBusy();
        if (!gallery.length) { els.stage.hidden = true; els.empty.hidden = false; }
        setStatus(d.error || ("job " + d.status), "error");
      })
      .catch(function (err) {
        // transient network hiccup — the queue never drops the job, so retry
        setStatus("connection hiccup — still polling (" + err.message + ")", "info");
        poll(jobId, 2000);
      });
  }, waitMs);
}

function finishDone(d) {
  stopBusy();
  var m = d.result || {};
  fetch(d.result_url, { headers: authHeaders() })
    .then(function (r) {
      if (!r.ok) throw new Error("result fetch HTTP " + r.status);
      return r.blob();
    })
    .then(function (blob) {
      var url = URL.createObjectURL(blob);
      els.image.src = url;
      els.stage.hidden = false;
      current = gallery.length;
      gallery.push({ url: url, seed: m.seed, width: m.width, height: m.height,
                     steps: m.steps, seconds: m.seconds, fast: m.fast,
                     prompt: els.prompt.value.trim() });
      els.meta.hidden = false;
      els.metaInfo.textContent = (m.fast ? "FAST · " : "") + m.width + "×" + m.height +
        " · " + m.steps + " steps · " + m.seconds + " s · seed " + m.seed;
      els.download.hidden = false;
      renderHistory();
      setStatus("done in " + m.seconds + " s (" + m.seconds_per_step + " s/step)", "info");
    })
    .catch(function (err) {
      setStatus("render finished but the image could not be fetched: " + err.message, "error");
      if (!gallery.length) { els.stage.hidden = true; els.empty.hidden = false; }
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
      els.metaInfo.textContent = (g.fast ? "FAST · " : "") + g.width + "×" + g.height +
        " · " + g.steps + " steps · " + g.seconds + " s · seed " + g.seed;
      renderHistory();
    });
    els.history.appendChild(b);
  });
}
</script>
</body>
</html>"""

app = FastAPI(title="Qwen-Image-2.1", version="2.0")


# ---------------------------------------------------------------- job store
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS jobs(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE,
                state TEXT NOT NULL,
                params TEXT NOT NULL,
                worker TEXT,
                error TEXT,
                requeues INTEGER DEFAULT 0,
                created_at REAL, started_at REAL, finished_at REAL,
                result_path TEXT)"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, seq)")


def submit_job(params: dict) -> tuple:
    job_id = uuid.uuid4().hex
    with db() as c:
        c.execute("BEGIN IMMEDIATE")
        c.execute(
            "INSERT INTO jobs(job_id,state,params,created_at) VALUES(?,?,?,?)",
            (job_id, "queued", json.dumps(params), time.time()),
        )
        pos = c.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE state='queued' AND seq>"
            "(SELECT seq FROM jobs WHERE job_id=?)",
            (job_id,),
        ).fetchone()["n"]
        c.execute("COMMIT")
    return job_id, pos  # pos = number of jobs ahead of this one (0-based)


def _need_gib(params: dict) -> float:
    pixels = (params["width"] / 32) * (params["height"] / 32)  # pipeline floors to /32
    mp = (pixels * 1024) / 1e6
    return TRANSIENT_GIB_PER_MP * mp + VRAM_SAFETY_GIB


def claim_next(worker: str, free_gib: float):
    """VRAM-aware first-fit claim: walk queued jobs in FIFO order and take the
    first whose estimated transient fits the currently-free VRAM MINUS the full
    estimated need of every job already rendering.

    Why minus-inflight: free VRAM is read before an in-flight render has
    allocated its transient (allocation ramps over seconds). Two dispatchers
    can both see "43 free" and both claim 2K jobs that collide at the peak —
    seen live 2026-10-06: 134 OOMs, card peaked at 138 GiB, when a 7x2K burst
    raced a user's web request. Accounting inflight jobs at their FULL need
    makes the gate conservative: a second 2K waits until the first finishes,
    while small jobs (whose need fits the remaining margin) still flow.

    Serialized by BEGIN IMMEDIATE across processes. Queued jobs never fail —
    they wait for a calm card."""
    with db() as c:
        c.execute("BEGIN IMMEDIATE")
        inflight = c.execute(
            "SELECT params FROM jobs WHERE state='rendering'"
        ).fetchall()
        inflight_need = 0.0
        for r in inflight:
            try:
                inflight_need += _need_gib(json.loads(r["params"]))
            except Exception:  # noqa: BLE001 — malformed row: assume worst
                inflight_need += 30.0
        effective_free = free_gib - inflight_need
        rows = c.execute(
            "SELECT seq,job_id,params FROM jobs WHERE state='queued' ORDER BY seq"
        ).fetchall()
        picked = None
        for row in rows:
            params = json.loads(row["params"])
            if _need_gib(params) <= effective_free:
                picked = row
                break
        if picked is None:
            c.execute("COMMIT")
            return None
        c.execute(
            "UPDATE jobs SET state='rendering',worker=?,started_at=? WHERE job_id=?",
            (worker, time.time(), picked["job_id"]),
        )
        c.execute("COMMIT")
    return dict(picked)


def finish_job(job_id: str, result_path: str, meta: dict) -> None:
    with db() as c:
        c.execute(
            "UPDATE jobs SET state='done',finished_at=?,result_path=?,error=? WHERE job_id=?",
            (time.time(), str(result_path), json.dumps(meta), job_id),
        )


def fail_job(job_id: str, message: str) -> None:
    with db() as c:
        c.execute(
            "UPDATE jobs SET state='error',finished_at=?,error=? WHERE job_id=?",
            (time.time(), message, job_id),
        )


def job_row(job_id: str):
    with db() as c:
        return c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()


def counts() -> dict:
    try:
        with db() as c:
            rows = c.execute(
                "SELECT state,COUNT(*) AS n FROM jobs GROUP BY state"
            ).fetchall()
        d = {r["state"]: r["n"] for r in rows}
        return {"queued": d.get("queued", 0), "rendering": d.get("rendering", 0)}
    except Exception:  # noqa: BLE001 — health must never raise
        return {"queued": -1, "rendering": -1}


def sweep() -> None:
    """Crash recovery + TTL cleanup, run periodically by any dispatcher.
    - rendering rows whose lease expired (a worker died mid-render) are
      requeued up to MAX_REQUEUES, then marked error. Queued jobs are never
      dropped — they wait forever until rendered or canceled.
    - done/error rows older than RESULT_TTL_S and their PNGs are deleted."""
    now = time.time()
    try:
        with db() as c:
            stale = c.execute(
                "SELECT job_id,requeues FROM jobs WHERE state='rendering' AND started_at<?",
                (now - RENDER_LEASE_S,),
            ).fetchall()
            for r in stale:
                if r["requeues"] + 1 >= MAX_REQUEUES:
                    fail_job(r["job_id"], "render did not complete (restart budget exceeded)")
                else:
                    c.execute(
                        "UPDATE jobs SET state='queued',worker=NULL,started_at=NULL,"
                        "requeues=requeues+1 WHERE job_id=?",
                        (r["job_id"],),
                    )
            old = c.execute(
                "SELECT job_id,result_path FROM jobs WHERE state IN ('done','error','canceled')"
                " AND finished_at<?",
                (now - RESULT_TTL_S,),
            ).fetchall()
        for r in old:
            if r["result_path"]:
                try:
                    pathlib.Path(r["result_path"]).unlink(missing_ok=True)
                except OSError:
                    pass
            with db() as c:
                c.execute("DELETE FROM jobs WHERE job_id=?", (r["job_id"],))
    except Exception as exc:  # noqa: BLE001 — a sweep failure must not kill dispatch
        print(f"dispatcher sweep error: {exc}", flush=True)


# ---------------------------------------------------------------- auth
def _consteq(a: str, b: str) -> bool:
    acc = 0
    for x, y in zip(a.encode(), b.encode()):
        acc |= x ^ y
    return acc == 0


async def require_token(authorization: str = Header(default="")) -> None:
    if not TOKEN:
        raise HTTPException(500, "QI21_API_KEY is not set on the server")
    expected = f"Bearer {TOKEN}"
    if len(authorization) != len(expected) or not _consteq(authorization, expected):
        raise HTTPException(401, "bad or missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


# ---------------------------------------------------------------- routes
@app.get("/ui")
@app.get("/")
async def ui():
    return HTMLResponse(UI_HTML)


@app.get("/health")
async def health():
    return {
        "status": "ok" if PIPE is not None else "loading",
        "model": "Qwen-Image-2.1",
        "dtype": "bfloat16",
        "workers": WORKERS,
        "worker_id": WORKER_ID,
        "resident_gib": round(torch.cuda.memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None,
        "uptime_s": round(time.time() - BOOT_T, 1),
        "busy": GPU_LOCK.locked(),
        "fast_ready": FAST_READY,
        **counts(),
    }


@app.get("/queue")
async def queue_view():
    """Lobby view (ComfyUI /queue precedent): global counts, no params."""
    return {"model": "Qwen-Image-2.1", **counts()}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="what to draw")
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=MAX_STEPS)
    cfg: float = Field(default=1.0, ge=1.0, le=8.0,
                       description="true_cfg_scale; 1.0 is the model's default. "
                                   "Above 1.0 runs the DiT twice per step (2x cost).")
    seed: int = Field(default=-1, description="-1 = random (resolved at submit, returned in status)")
    negative_prompt: str = Field(default="",
                                 description="only used when cfg > 1 — the pipeline "
                                             "ignores it at true_cfg_scale=1")
    fast: bool = Field(default=False,
                       description="Turbo8 distilled lane: 8 steps, cfg forced 1.0, "
                                   "~1.5s per 1024px image. Cost: dense text and complex "
                                   "edits degrade (text exact-match 95% -> 75%).")


def _resolve_params(req: GenerateRequest) -> dict:
    if req.width * req.height > MAX_PIXELS:
        raise HTTPException(400, f"too many pixels: cap is {MAX_PIXELS}")
    if req.fast and not FAST_READY:
        raise HTTPException(503, "fast mode unavailable — the Turbo8 LoRA did not load at boot")
    steps = 8 if req.fast else req.steps
    cfg = 1.0 if req.fast else req.cfg
    seed = req.seed if req.seed >= 0 else secrets.randbits(31)
    negative = req.negative_prompt if (req.negative_prompt and cfg > 1.0) else ""
    # negative_prompt is a silent no-op at cfg=1 per the diffusers docs — the
    # API drops it rather than pretending it was used.
    return {"prompt": req.prompt, "width": req.width, "height": req.height,
            "steps": steps, "cfg": cfg, "seed": seed,
            "negative_prompt": negative, "fast": bool(req.fast)}


def _envelope(job_id: str, pos: int) -> JSONResponse:
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "status": "queued",
        "queue_position": pos,                      # 0-based, fal-style
        "status_url": f"/jobs/{job_id}",
        "result_url": f"/jobs/{job_id}/result",
        "cancel_url": f"/jobs/{job_id}/cancel",
    })


@app.post("/generate", dependencies=[Depends(require_token)])
async def generate(req: GenerateRequest):
    """Submit and return immediately. Never rejects for capacity — the queue
    is unbounded; the render happens when a worker frees up."""
    params = _resolve_params(req)
    job_id, pos = submit_job(params)
    return _envelope(job_id, pos)


def _wait_done(job_id: str) -> bool:
    deadline = time.time() + SYNC_WAIT_S
    while time.time() < deadline:
        row = job_row(job_id)
        if row is not None and row["state"] in ("done", "error", "canceled"):
            return True
        time.sleep(0.4)
    return False


def _openai_shape(row) -> dict:
    meta = json.loads(row["error"]) if row["error"] else {}
    with open(row["result_path"], "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return {
        "created": int(row["finished_at"] or time.time()),
        "data": [{"b64_json": b64, "url": None, "revised_prompt": None}],
        "seed": meta.get("seed"), "width": meta.get("width"),
        "height": meta.get("height"), "steps": meta.get("steps"),
        "seconds": meta.get("seconds"), "seconds_per_step": meta.get("seconds_per_step"),
        "fast": meta.get("fast", False),
        "job_id": row["job_id"],
    }


@app.post("/v1/images/generations", dependencies=[Depends(require_token)])
async def generate_openai(req: GenerateRequest):
    """OpenAI-shaped clients need a synchronous body. Submit, wait up to
    QI21_SYNC_WAIT_S, answer with the OpenAI shape — or the 202 envelope if the
    queue is deeper than the wait. The queue can delay, never reject."""
    params = _resolve_params(req)
    job_id, pos = submit_job(params)
    done = await asyncio.to_thread(_wait_done, job_id)
    if not done:
        return _envelope(job_id, queue_position(job_id) if queue_position(job_id) is not None else pos)
    row = job_row(job_id)
    if row["state"] == "error":
        return JSONResponse(status_code=500, content={"error": row["error"]})
    if row["state"] == "canceled":
        return JSONResponse(status_code=400, content={"error": "canceled"})
    return _openai_shape(row)


def _job_status(row) -> dict:
    meta = json.loads(row["error"]) if (row["state"] == "done" and row["error"]) else {}
    out = {
        "job_id": row["job_id"],
        "status": row["state"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"] if row["state"] == "error" else None,
        "result_url": f"/jobs/{row['job_id']}/result" if row["state"] == "done" else None,
        "result": meta if row["state"] == "done" else None,
    }
    if row["state"] == "queued":
        with db() as c:
            n = c.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE state='queued' AND seq<?",
                (row["seq"],),
            ).fetchone()["n"]
        out["queue_position"] = n  # 0-based; the UI displays n+1
    return out


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
async def job_status(job_id: str):
    row = job_row(job_id)
    if row is None:
        raise HTTPException(404, "unknown job id")
    return _job_status(row)


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_token)])
async def job_result(job_id: str):
    row = job_row(job_id)
    if row is None:
        raise HTTPException(404, "unknown job id")
    if row["state"] != "done" or not row["result_path"]:
        raise HTTPException(409, f"job is {row['state']}, no result yet")
    if not pathlib.Path(row["result_path"]).exists():
        raise HTTPException(410, "result expired")
    return FileResponse(row["result_path"], media_type="image/png",
                        filename=f"qwen21_{job_id}.png")


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_token)])
async def job_cancel(job_id: str):
    with db() as c:
        cur = c.execute(
            "UPDATE jobs SET state='canceled',finished_at=? WHERE job_id=? AND state='queued'",
            (time.time(), job_id),
        )
    if cur.rowcount:
        return {"job_id": job_id, "status": "canceled"}
    raise HTTPException(409, "cancel only works before rendering starts")


# ---------------------------------------------------------------- dispatcher
def _render_sync(params: dict, job_id: str):
    """Runs in a worker thread while GPU_LOCK is held. Returns (png_path, meta)."""
    t0 = time.time()
    kwargs = dict(
        prompt=params["prompt"], width=params["width"], height=params["height"],
        num_inference_steps=params["steps"], true_cfg_scale=params["cfg"],
        generator=torch.Generator("cuda").manual_seed(params["seed"]),
        # Pinned: toggling this does not reproduce images bit-for-bit in
        # reduced precision — pin it so a given (prompt, seed, steps) is
        # reproducible across the fleet.
        use_kv_cache=True,
    )
    if params.get("negative_prompt"):
        kwargs["negative_prompt"] = params["negative_prompt"]
    fast = bool(params.get("fast"))
    if fast:
        PIPE.scheduler = FAST_SCHEDULER
    try:
        image = PIPE(**kwargs).images[0]
    finally:
        if fast:
            PIPE.scheduler = BASE_SCHEDULER
    dt = time.time() - t0
    meta = {"seed": params["seed"], "width": params["width"], "height": params["height"],
            "steps": params["steps"], "seconds": round(dt, 2),
            "seconds_per_step": round(dt / params["steps"], 3), "fast": fast}
    # PNG encoding (compress_level=1: transport, not archival) happens after
    # timing was taken, so `seconds` stays an honest GPU number.
    out = RESULTS / f"{job_id}.png"
    image.save(out, format="PNG", compress_level=1)
    return out, meta


async def dispatcher():
    me = WORKER_ID
    i = 0
    row = None
    print(f"dispatcher {me} up — claiming jobs", flush=True)
    while True:
        try:
            torch.cuda.empty_cache()   # release our dead blocks so big jobs can claim
            free_gib = torch.cuda.mem_get_info()[0] / 2**30
            row = claim_next(me, free_gib)
            if row is None:
                if i % 400 == 0:
                    sweep()  # ~every 100s of idle time
                await asyncio.sleep(0.25)
            else:
                params = json.loads(row["params"])
                print(f"[{me}] rendering {row['job_id']} "
                      f"({params['width']}x{params['height']} x{params['steps']}"
                      f"{' fast' if params.get('fast') else ''})", flush=True)
                async with GPU_LOCK:
                    path, meta = await asyncio.to_thread(_render_sync, params, row["job_id"])
                finish_job(row["job_id"], str(path), meta)
                print(f"[{me}] done {row['job_id']} in {meta['seconds']}s", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            # VRAM contention between replicas: shrink this worker's cached
            # blocks, put the job back in the queue, and retry when the other
            # workers release their transients. The job is never lost — the
            # queue holds it and a later attempt succeeds once memory frees.
            torch.cuda.empty_cache()
            rq = 0
            if row is not None:
                with db() as c:
                    r = c.execute(
                        "SELECT requeues FROM jobs WHERE job_id=?", (row["job_id"],)
                    ).fetchone()
                    rq = ((r["requeues"] if r else 0) or 0) + 1
                    if rq >= MAX_REQUEUES:
                        fail_job(row["job_id"], f"repeated CUDA OOM: {exc}")
                    else:
                        c.execute(
                            "UPDATE jobs SET state='queued',worker=NULL,started_at=NULL,"
                            "requeues=? WHERE job_id=?",
                            (rq, row["job_id"]),
                        )
                print(f"[{me}] OOM on {row['job_id']} — requeued (attempt {rq}/{MAX_REQUEUES})",
                      flush=True)
            row = None
            await asyncio.sleep(3)
        except Exception as exc:  # noqa: BLE001 — a failed job must not kill dispatch
            try:
                if row is not None:
                    fail_job(row["job_id"], f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            print(f"dispatcher error: {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(1)
        i += 1


# ---------------------------------------------------------------- startup
def load() -> None:
    """Load the pipeline. Runs in each worker's startup hook."""
    global PIPE, BASE_SCHEDULER, FAST_SCHEDULER, FAST_READY
    from diffusers import QwenImage21Pipeline

    print(f"loading {WORK} -> cuda (bfloat16, no offload)", flush=True)
    t0 = time.time()
    PIPE = QwenImage21Pipeline.from_pretrained(str(WORK), dtype=torch.bfloat16).to("cuda")
    BASE_SCHEDULER = PIPE.scheduler
    gib = torch.cuda.memory_allocated() / 2**30
    print(f"MODEL_READY in {time.time()-t0:.1f}s — {gib:.1f} GiB resident", flush=True)

    # Fast lane: Turbo8 (r128, 8 steps, T2I + editing + RGBA + up to 2K).
    # Requirements from its model card, not negotiable: the LoRA plus a
    # scheduler with shift_terminal unset. Strictly opt-in; base stays default.
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


@asynccontextmanager
async def lifespan(_app):
    global BOOT_T
    BOOT_T = time.time()
    if not TOKEN:
        raise RuntimeError("QI21_API_KEY is not set — refusing to serve unauthenticated")
    init_db()
    await asyncio.to_thread(load)
    task = asyncio.create_task(dispatcher())
    yield
    task.cancel()


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    import uvicorn

    if not torch.cuda.is_available():
        raise SystemExit("!! no CUDA — this server needs the GPU Studio")
    # Import-string form is REQUIRED for workers>1: uvicorn spawns worker
    # processes that import this module and run the lifespan above, so every
    # worker gets its own pipeline and its own dispatcher.
    print(f"uvicorn up: {WORKERS} workers, queue=sqlite-wal, sync wait {SYNC_WAIT_S}s", flush=True)
    uvicorn.run("qi21_server:app", host="0.0.0.0", port=PORT, workers=WORKERS, log_level="warning")
QISERVER_EOF
echo "wrote qi21_server.py from the embedded copy ($(wc -c < qi21_server.py) bytes)"

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
QI21_API_KEY="$QI21_API_KEY" QI21_WORK="$WORK" QI21_WORKERS="$WORKERS" \
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
echo "#  workers: $WORKERS (queue in jobs.db — jobs survive restarts)"
echo "############################################################"
