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

# The API reference page, embedded verbatim from the kit's qi21_docs.html
# (byte-identity is verified by cell_06_serve.sh's build step). This is the
# ONLY human-facing surface: the service is API-first — `/` is the docs, and
# every other route is JSON.
DOCS_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen-Image-2.1 API — Reference</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0b0c0f;
    --bg-raised: #101218;
    --surface: #14161c;
    --border: #232733;
    --border-soft: #1a1d26;
    --text: #e7e9ef;
    --text-2: #a7afc0;
    --muted: #7c8598;
    --brand: #e8a33d;
    --brand-ink: #171003;
    --get: #3ecf8e;
    --post: #5b9dff;
    --danger: #e5645f;
    --code-bg: #0d0f14;
    --radius: 10px;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    --sans: ui-sans-serif, system-ui, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 15px/1.65 var(--sans);
    -webkit-font-smoothing: antialiased;
  }
  ::selection { background: rgba(232,163,61,.28); }
  a { color: var(--brand); text-decoration: none; }
  a:hover { text-decoration: underline; text-underline-offset: 3px; }
  :focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; border-radius: 4px; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-thumb { background: #2a2e3a; border-radius: 5px; border: 2px solid var(--bg); }
  ::-webkit-scrollbar-track { background: transparent; }

  /* ---------- top bar ---------- */
  .top {
    position: sticky; top: 0; z-index: 50;
    display: flex; align-items: center; gap: 14px;
    padding: 0 22px; height: 56px;
    background: rgba(11,12,15,.86);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border-soft);
  }
  .mark { width: 22px; height: 22px; flex: 0 0 auto; }
  .top .name { font-weight: 700; font-size: 14.5px; letter-spacing: -.01em; }
  @media (max-width: 700px) {
    .baseurl { display: none; }
    .top .name { font-size: 13.5px; white-space: nowrap; }
  }
  .top .ver {
    font: 600 11px var(--mono); color: var(--brand);
    border: 1px solid color-mix(in srgb, var(--brand) 40%, transparent);
    border-radius: 999px; padding: 2px 8px;
  }
  .top .spacer { flex: 1; }
  .baseurl {
    display: flex; align-items: center; gap: 8px;
    font: 12px var(--mono); color: var(--text-2);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 10px;
    max-width: 44vw; overflow: hidden;
  }
  .baseurl span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .baseurl button {
    all: unset; cursor: pointer; color: var(--muted);
    display: inline-flex; padding: 2px;
  }
  .baseurl button:hover { color: var(--brand); }
  .baseurl button svg { width: 13px; height: 13px; }

  /* ---------- layout ---------- */
  .shell { display: grid; grid-template-columns: 256px minmax(0,1fr); }
  .shell > * { min-width: 0; }   /* grid items must be allowed to shrink below min-content */
  main > * { min-width: 0; }
  @media (max-width: 960px) { .shell { grid-template-columns: 1fr; } }

  nav.side {
    position: sticky; top: 56px;
    height: calc(100vh - 56px);
    overflow-y: auto;
    border-right: 1px solid var(--border-soft);
    padding: 22px 14px 40px;
  }
  @media (max-width: 960px) {
    nav.side {
      position: static; height: auto;
      border-right: 0; border-bottom: 1px solid var(--border-soft);
    }
  }
  nav .group { margin-bottom: 22px; }
  .hamburger {
    display: none;
    appearance: none; -webkit-appearance: none;
    background: transparent; border: 0;
    cursor: pointer; color: var(--text);
    align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 8px;
  }
  .hamburger svg { width: 20px; height: 20px; }
  .hamburger:hover { color: var(--brand); background: var(--surface); }
  .hamburger[aria-expanded="true"] { color: var(--brand); background: var(--surface); }
  .backdrop {
    position: fixed; inset: 0; z-index: 80;
    background: rgba(5,6,9,.55);
    opacity: 0; pointer-events: none;
    transition: opacity .2s ease-out;
  }
  .backdrop.show { opacity: 1; pointer-events: auto; }
  @media (max-width: 960px) {
    .hamburger { display: inline-flex; }
    nav.side {
      position: fixed; top: 0; bottom: 0; left: 0;
      width: min(82vw, 320px); height: 100dvh;
      z-index: 90;
      background: var(--bg-raised);
      border-right: 1px solid var(--border);
      transform: translateX(-103%);
      transition: transform .22s ease-out;
      padding-top: 18px;
    }
    nav.side.open { transform: none; box-shadow: 12px 0 40px rgba(0,0,0,.45); }
  }
  nav .group h4 {
    margin: 0 0 6px; padding: 0 10px;
    font-size: 11.5px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: var(--muted);
  }
  nav a {
    display: block; padding: 5px 10px; margin: 1px 0;
    border-radius: 7px; color: var(--text-2); font-size: 13px;
    border-left: 2px solid transparent;
  }
  nav a:hover { color: var(--text); background: var(--surface); text-decoration: none; }
  nav a.active { color: var(--brand); background: color-mix(in srgb, var(--brand) 8%, transparent); border-left-color: var(--brand); }

  main { padding: 44px 48px 90px; max-width: 800px; }
  @media (max-width: 960px) { main { padding: 32px 22px 70px; } }

  /* ---------- typography ---------- */
  h1 { font-size: 30px; font-weight: 750; letter-spacing: -.025em; margin: 0 0 10px; }
  .lede { color: var(--text-2); font-size: 16px; margin: 0 0 8px; max-width: 68ch; }
  h2 {
    font-size: 21px; font-weight: 700; letter-spacing: -.02em;
    margin: 54px 0 6px; padding-top: 26px;
    border-top: 1px solid var(--border-soft);
  }
  h2:first-of-type { border-top: 0; }
  h3 { font-size: 15.5px; font-weight: 650; margin: 26px 0 4px; }
  p { color: var(--text-2); max-width: 72ch; margin: 8px 0; }
  p strong, li strong { color: var(--text); font-weight: 600; }
  code.inl {
    font: 12.5px var(--mono);
    background: var(--surface); border: 1px solid var(--border-soft);
    border-radius: 5px; padding: 1px 5px; color: #d8bc8a;
  }
  section { scroll-margin-top: 76px; }

  /* ---------- endpoint blocks ---------- */
  .ep {
    background: var(--bg-raised);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 18px 20px 20px;
    margin: 14px 0 8px;
  }
  .ep-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .method {
    font: 700 11px var(--mono); letter-spacing: .06em;
    border-radius: 6px; padding: 3px 8px;
  }
  .method.post { color: var(--post); background: rgba(91,157,255,.13); border: 1px solid rgba(91,157,255,.35); }
  .method.get  { color: var(--get);  background: rgba(62,207,142,.12); border: 1px solid rgba(62,207,142,.32); }
  .ep-path { font: 600 14px var(--mono); color: var(--text); }
  .ep-desc { color: var(--text-2); font-size: 13.5px; margin: 8px 0 0; }

  /* ---------- tables ---------- */
  table {
    width: 100%; border-collapse: collapse; margin: 12px 0 6px; font-size: 13.5px;
    display: block; overflow-x: auto;   /* narrow screens scroll the table, not the page */
  }
  th {
    text-align: left; font-size: 11px; font-weight: 700;
    letter-spacing: .07em; text-transform: uppercase; color: var(--muted);
    padding: 7px 12px 7px 0; border-bottom: 1px solid var(--border);
  }
  td { padding: 8px 12px 8px 0; border-bottom: 1px solid var(--border-soft); vertical-align: top; color: var(--text-2); }
  td:first-child { white-space: nowrap; }
  td code, .req { font: 12px var(--mono); color: #d8bc8a; }
  .type { color: var(--muted); font: 12px var(--mono); }
  .default { color: var(--get); font: 12px var(--mono); }
  .num { font-variant-numeric: tabular-nums; }

  /* ---------- code blocks ---------- */
  .code {
    position: relative;
    background: var(--code-bg);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    margin: 12px 0;
    overflow: hidden;
  }
  .code .lang {
    display: flex; align-items: center; justify-content: space-between;
    padding: 7px 12px;
    border-bottom: 1px solid var(--border-soft);
    font: 600 11px var(--mono); letter-spacing: .05em; color: var(--muted);
  }
  .code pre {
    margin: 0; padding: 13px 16px;
    overflow-x: auto;
    font: 12.5px/1.6 var(--mono);
    color: #cdd3e0;
  }
  .copy {
    all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 5px;
    color: var(--muted); font: 600 11px var(--sans); padding: 3px 6px; border-radius: 5px;
  }
  .copy:hover { color: var(--brand); background: var(--surface); }
  .copy svg { width: 12px; height: 12px; }
  .copy.ok { color: var(--get); }

  /* ---------- callouts ---------- */
  .note {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px; margin: 14px 0;
    font-size: 13.5px; color: var(--text-2);
    max-width: 72ch;
  }
  .note b { color: var(--text); }
  .note.brand { border-color: color-mix(in srgb, var(--brand) 35%, var(--border)); }

  .flow {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    font: 600 12px var(--mono); margin: 12px 0;
  }
  .flow .st { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 4px 9px; }
  .flow .st.q { color: var(--brand); }
  .flow .st.r { color: var(--post); }
  .flow .st.d { color: var(--get); }
  .flow .st.e { color: var(--danger); }
  .flow .arr { color: var(--muted); }

  footer {
    margin-top: 70px; padding-top: 22px;
    border-top: 1px solid var(--border-soft);
    color: var(--muted); font-size: 12.5px;
  }
</style>
</head>
<body>

<header class="top">
  <button type="button" class="hamburger" aria-label="Open contents" aria-expanded="false" aria-controls="toc-groups">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
  </button>
  <svg class="mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <rect x="2.5" y="2.5" width="19" height="19" rx="5.5" stroke="#e8a33d" stroke-width="1.6"/>
    <circle cx="12" cy="12" r="4.2" stroke="#e8a33d" stroke-width="1.6"/>
    <circle cx="12" cy="12" r="1.3" fill="#e8a33d"/>
  </svg>
  <span class="name">Qwen-Image-2.1 API</span>
  <span class="ver">v2.0</span>
  <span class="spacer"></span>
  <span class="baseurl" id="base">
    <span>https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev</span>
    <button type="button" data-copy="https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev" aria-label="Copy base URL">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>
    </button>
  </span>
</header>
<div class="backdrop" aria-hidden="true"></div>

<div class="shell">
<nav class="side" id="toc-groups" aria-label="Sections">
  <div class="groups">
  <div class="group">
    <h4>Getting started</h4>
    <a href="#overview">Overview</a>
    <a href="#quickstart">Quickstart</a>
    <a href="#auth">Authentication</a>
  </div>
  <div class="group">
    <h4>Endpoints</h4>
    <a href="#post-generate">POST /generate</a>
    <a href="#post-openai">POST /v1/images/generations</a>
    <a href="#get-job">GET /jobs/&#123;id&#125;</a>
    <a href="#get-result">GET /jobs/&#123;id&#125;/result</a>
    <a href="#post-cancel">POST /jobs/&#123;id&#125;/cancel</a>
    <a href="#get-queue">GET /queue</a>
    <a href="#get-health">GET /health</a>
  </div>
  <div class="group">
    <h4>Behavior</h4>
    <a href="#queue-behavior">Queue semantics</a>
    <a href="#fast-lane">Fast lane</a>
    <a href="#errors">Error codes</a>
  </div>
  <div class="group">
    <h4>Reference</h4>
    <a href="#limits">Limits &amp; config</a>
    <a href="#performance">Measured performance</a>
  </div>
  </div>
</nav>

<main>

<h1>Qwen-Image-2.1 API</h1>
<p class="lede">Full-precision text-to-image generation. Submit a prompt, hold your place in a
queue that never rejects a request, and fetch your image when a worker frees up.
Three GPU workers, one shared queue, zero rejections.</p>

<!-- ============================================================ overview -->
<section id="overview">
<h2>Overview</h2>
<p>The service runs the complete, unquantized Qwen-Image-2.1 model (33.1 GB BF16)
on an NVIDIA H200 across three worker processes. Every generation request is
admitted to a durable FIFO queue — at any traffic level the response is either a
finished image or a queue position, never a rejection.</p>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Model</td><td>Qwen-Image-2.1 — 7B DiT + Qwen3-VL 8B encoder, full BF16</td></tr>
<tr><td>Resolutions</td><td>64–2048 px per side, multiples of 32 (native 2K supported)</td></tr>
<tr><td>Queue</td><td class="num">Unbounded FIFO — requests wait, never fail</td></tr>
<tr><td>Workers</td><td class="num">3 concurrent renders</td></tr>
<tr><td>Result retention</td><td class="num">24 hours after completion</td></tr>
<tr><td>License</td><td>Qwen Research — non-commercial research and evaluation</td></tr>
</table>
<div class="note"><b>The golden rule of this API:</b> a submit can only fail validation
(bad key, bad parameters). Capacity is never a failure — when all workers are busy,
you get a queue position and your turn comes in order.</div>
</section>

<!-- ============================================================ quickstart -->
<section id="quickstart">
<h2>Quickstart</h2>
<p>Three calls: submit, poll, download.</p>

<div class="code"><div class="lang"><span>1 · submit</span>
<button type="button" class="copy" data-copy="curl -s -X POST https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/generate -H 'Authorization: Bearer YOUR_KEY' -H 'Content-Type: application/json' -d '{&quot;prompt&quot;:&quot;a lighthouse in fog&quot;,&quot;width&quot;:1024,&quot;height&quot;:1024,&quot;steps&quot;:28,&quot;seed&quot;:7}'">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>copy</button></div>
<pre>curl -s -X POST https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/generate \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a lighthouse in fog","width":1024,"height":1024,"steps":28,"seed":7}'

# → 202
# {
#   "job_id": "e7d850545110…",
#   "status": "queued",
#   "queue_position": 0,                       ← 0-based; you are #1
#   "status_url":  "/jobs/e7d850545110…",
#   "result_url":  "/jobs/e7d850545110…/result",
#   "cancel_url":  "/jobs/e7d850545110…/cancel"
# }</pre></div>

<div class="code"><div class="lang"><span>2 · poll until done</span>
<button type="button" class="copy" data-copy="curl -s https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/jobs/JOB_ID -H 'Authorization: Bearer YOUR_KEY'">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>copy</button></div>
<pre>curl -s https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/jobs/JOB_ID \
  -H "Authorization: Bearer YOUR_KEY"

# while waiting:
# { "status": "queued",    "queue_position": 1,  … }
# { "status": "rendering", … }
# when finished:
# { "status": "done",
#   "result": { "seed": 7, "width": 1024, "height": 1024,
#               "steps": 28, "seconds": 4.3,
#               "seconds_per_step": 0.154, "fast": false },
#   "result_url": "/jobs/JOB_ID/result" }</pre></div>

<div class="code"><div class="lang"><span>3 · fetch the PNG</span>
<button type="button" class="copy" data-copy="curl -s https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/jobs/JOB_ID/result -H 'Authorization: Bearer YOUR_KEY' -o image.png">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>copy</button></div>
<pre>curl -s https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/jobs/JOB_ID/result \
  -H "Authorization: Bearer YOUR_KEY" -o image.png</pre></div>
</section>

<!-- ============================================================ auth -->
<section id="auth">
<h2>Authentication</h2>
<p>Every endpoint except <code class="inl">/health</code> and <code class="inl">/queue</code>
requires a bearer token. Send it in the <code class="inl">Authorization</code> header;
a missing or wrong token returns <b>401</b>.</p>
<div class="note"><b>Browser clients:</b> requests made from JavaScript must also send
the <code class="inl">ngrok-skip-browser-warning: true</code> header to bypass the
tunnel's first-visit interstitial. Server-side clients (curl, SDKs) don't need it.</div>
</section>

<!-- ============================================================ POST /generate -->
<section id="post-generate">
<h2>Submit a generation</h2>
<div class="ep">
  <div class="ep-head"><span class="method post">POST</span><span class="ep-path">/generate</span></div>
  <p class="ep-desc">Adds a job to the queue and returns immediately with its position.
  The queue is unbounded — this endpoint does not have a capacity-based failure mode.</p>
</div>

<h3>Request body</h3>
<table>
<tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
<tr><td><code>prompt</code></td><td class="type">string</td><td class="default">required</td>
  <td>What to draw. Minimum length 1.</td></tr>
<tr><td><code>width</code></td><td class="type">int</td><td class="default">1024</td>
  <td>64–2048. Must be a multiple of 32 (values are floored).</td></tr>
<tr><td><code>height</code></td><td class="type">int</td><td class="default">1024</td>
  <td>Same rules as width. Total pixels are capped at 2048×2048.</td></tr>
<tr><td><code>steps</code></td><td class="type">int</td><td class="default">20</td>
  <td>1–50. Cost scales linearly; more steps refine detail. 28 is a good balance.</td></tr>
<tr><td><code>cfg</code></td><td class="type">float</td><td class="default">1.0</td>
  <td>Guidance scale. The model is guidance-free at 1.0 — that is the intended
  full-quality setting. Above 1.0 the model runs twice per step (2× cost).</td></tr>
<tr><td><code>seed</code></td><td class="type">int</td><td class="default">-1</td>
  <td>-1 picks a random seed, which is returned in the job status. Same seed +
  same parameters = same image.</td></tr>
<tr><td><code>negative_prompt</code></td><td class="type">string</td><td class="default">""</td>
  <td>Only used when <code>cfg</code> &gt; 1 — the pipeline ignores it at the default
  setting, so the server drops it rather than pretending it was applied.</td></tr>
<tr><td><code>fast</code></td><td class="type">bool</td><td class="default">false</td>
  <td>Opt-in distilled lane: forces 8 steps and cfg 1.0, roughly 4× faster.
  Trade-off: dense text and complex compositions degrade. See
  <a href="#fast-lane">Fast lane</a>.</td></tr>
</table>

<h3>Response — 202 Accepted</h3>
<div class="code"><div class="lang"><span>application/json</span>
<button type="button" class="copy" data-copy='{"job_id":"e7d850545110","status":"queued","queue_position":0,"status_url":"/jobs/e7d850545110","result_url":"/jobs/e7d850545110/result","cancel_url":"/jobs/e7d850545110/cancel"}'>
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>copy</button></div>
<pre>{
  "job_id": "e7d850545110…",
  "status": "queued",
  "queue_position": 0,
  "status_url": "/jobs/e7d850545110…",
  "result_url": "/jobs/e7d850545110…/result",
  "cancel_url": "/jobs/e7d850545110…/cancel"
}</pre></div>
<p><code class="inl">queue_position</code> is a snapshot at submit time (0-based —
position 0 means you are next). Re-poll <code class="inl">status_url</code> for the
live value.</p>
</section>

<!-- ============================================================ OpenAI route -->
<section id="post-openai">
<h2>OpenAI-compatible endpoint</h2>
<div class="ep">
  <div class="ep-head"><span class="method post">POST</span><span class="ep-path">/v1/images/generations</span></div>
  <p class="ep-desc">Same parameters as <code class="inl">/generate</code>, but the call
  stays open for up to 120 seconds and answers with the OpenAI images shape — so the
  official OpenAI SDK works unmodified. If the queue is deeper than the wait, it
  returns the same 202 envelope as <code class="inl">/generate</code> and you fall back
  to polling.</p>
</div>
<div class="code"><div class="lang"><span>python — official SDK</span>
<button type="button" class="copy" data-copy='from openai import OpenAI

client = OpenAI(
    base_url="https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/v1",
    api_key="YOUR_KEY",
)
img = client.images.generate(model="qwen-image-2.1", prompt="a lighthouse in fog")
import base64
open("out.png", "wb").write(base64.b64decode(img.data[0].b64_json))'>
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>copy</button></div>
<pre>from openai import OpenAI

client = OpenAI(
    base_url="https://pseudoasymmetric-unbodied-sabine.ngrok-free.dev/v1",
    api_key="YOUR_KEY",
)
img = client.images.generate(model="qwen-image-2.1", prompt="a lighthouse in fog")
open("out.png", "wb").write(base64.b64decode(img.data[0].b64_json))</pre></div>
<div class="note">Extras beyond the OpenAI schema arrive as top-level keys:
<code class="inl">seed</code>, <code class="inl">seconds</code>,
<code class="inl">seconds_per_step</code>, <code class="inl">fast</code>,
<code class="inl">job_id</code>. OpenAI clients ignore them; you can use them.</div>
</section>

<!-- ============================================================ jobs status -->
<section id="get-job">
<h2>Job status</h2>
<div class="ep">
  <div class="ep-head"><span class="method get">GET</span><span class="ep-path">/jobs/{job_id}</span></div>
  <p class="ep-desc">Live state of one job. Poll this while waiting.</p>
</div>
<h3>Lifecycle</h3>
<div class="flow">
  <span class="st q">queued</span><span class="arr">→</span>
  <span class="st r">rendering</span><span class="arr">→</span>
  <span class="st d">done</span>
</div>
<div class="flow">
  <span class="st q">queued</span><span class="arr">→</span>
  <span class="st e">canceled</span>
  <span style="color:var(--muted);font-weight:400">(cancel only works in this state)</span>
</div>
<h3>Response fields</h3>
<table>
<tr><th>Field</th><th>Present</th><th>Description</th></tr>
<tr><td><code>status</code></td><td class="type">always</td><td><code>queued</code> · <code>rendering</code> · <code>done</code> · <code>error</code> · <code>canceled</code></td></tr>
<tr><td><code>queue_position</code></td><td class="type">while queued</td><td class="num">0-based jobs ahead of you, recomputed on every poll</td></tr>
<tr><td><code>result</code></td><td class="type">when done</td><td><code>{seed, width, height, steps, seconds, seconds_per_step, fast}</code></td></tr>
<tr><td><code>result_url</code></td><td class="type">when done</td><td>Where to fetch the PNG</td></tr>
<tr><td><code>error</code></td><td class="type">when error</td><td>What went wrong (e.g. OOM after retries)</td></tr>
</table>
</section>

<!-- ============================================================ result -->
<section id="get-result">
<h2>Fetch the image</h2>
<div class="ep">
  <div class="ep-head"><span class="method get">GET</span><span class="ep-path">/jobs/{job_id}/result</span></div>
  <p class="ep-desc">Returns the finished image as <b>PNG bytes</b>. Available for 24
  hours after completion. <b>409</b> if the job hasn't finished, <b>410</b> if the
  result expired, <b>404</b> for an unknown id.</p>
</div>
</section>

<!-- ============================================================ cancel -->
<section id="post-cancel">
<h2>Cancel a job</h2>
<div class="ep">
  <div class="ep-head"><span class="method post">POST</span><span class="ep-path">/jobs/{job_id}/cancel</span></div>
  <p class="ep-desc">Removes a job while it is still waiting. Returns <b>200</b> on
  success, <b>409</b> if rendering already started. A canceled slot frees up for the
  next job in line.</p>
</div>
</section>

<!-- ============================================================ queue + health -->
<section id="get-queue">
<h2>Queue lobby</h2>
<div class="ep">
  <div class="ep-head"><span class="method get">GET</span><span class="ep-path">/queue</span></div>
  <p class="ep-desc">Global counts, no authentication. Handy for a status badge.</p>
</div>
<div class="code"><div class="lang"><span>response</span>
<button type="button" class="copy" data-copy='{"model":"Qwen-Image-2.1","queued":2,"rendering":3}'>
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>copy</button></div>
<pre>{ "model": "Qwen-Image-2.1", "queued": 2, "rendering": 3 }</pre></div>
</section>

<section id="get-health">
<h2>Health</h2>
<div class="ep">
  <div class="ep-head"><span class="method get">GET</span><span class="ep-path">/health</span></div>
  <p class="ep-desc">Per-worker view: this process's identity, VRAM residency, queue
  depth, and whether the fast lane is loaded. No authentication.</p>
</div>
<div class="code"><div class="lang"><span>response</span>
<button type="button" class="copy" data-copy='{"status":"ok","model":"Qwen-Image-2.1","dtype":"bfloat16","workers":3,"worker_id":"worker-84543","resident_gib":30.9,"uptime_s":459.9,"busy":false,"fast_ready":true,"queued":0,"rendering":0}'>
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>copy</button></div>
<pre>{
  "status": "ok", "model": "Qwen-Image-2.1", "dtype": "bfloat16",
  "workers": 3, "worker_id": "worker-84543",
  "resident_gib": 30.9, "uptime_s": 459.9,
  "busy": false, "fast_ready": true,
  "queued": 0, "rendering": 0
}</pre></div>
</section>

<!-- ============================================================ queue behavior -->
<section id="queue-behavior">
<h2>Queue semantics</h2>
<p>The queue is the contract. These properties are enforced by design and have been
verified under load:</p>
<table>
<tr><th>Property</th><th>Behavior</th></tr>
<tr><td>Never rejects</td><td>No capacity-based errors exist. An unbounded number of jobs can wait; each is ~300 bytes of state.</td></tr>
<tr><td>Strict FIFO</td><td>The oldest queued job takes the first free worker. Positions are recomputed on every poll — always fresh, never drift.</td></tr>
<tr><td>VRAM-aware</td><td>A job only starts when the card genuinely has room for its size. Large renders (2K) wait for a calm card instead of failing; small renders behind them keep flowing.</td></tr>
<tr><td>Crash-proof</td><td>Jobs live in a durable store. A worker that dies mid-render has its job requeued automatically (up to 6 attempts). Pending jobs survive server and Studio restarts.</td></tr>
<tr><td>Disconnect-proof</td><td>Closing your connection changes nothing. Poll again later with the same job id — the result waits up to 24 hours.</td></tr>
<tr><td>Serializable big renders</td><td>At full settings, 2K renders run one at a time by design; smaller renders can slot into the remaining memory alongside them.</td></tr>
</table>
</section>

<!-- ============================================================ fast lane -->
<section id="fast-lane">
<h2>Fast lane</h2>
<p>Setting <code class="inl">"fast": true</code> routes your request to a distilled
Turbo8 adapter: 8 steps instead of 20–50, roughly <b>4× faster</b> (~1.5 s per
1024² image). The server forces 8 steps and cfg 1.0 on this lane and ignores
contradicting parameters.</p>
<p>The trade-off is real and measured: overall prompt adherence holds, but
<b>dense or small text inside the image degrades</b> (character accuracy drops from
~99% to ~95%, exact text match from 95% to 75%). Use fast mode for exploration and
drafts; use the base lane for final renders and anything with legible lettering.</p>
</section>

<!-- ============================================================ errors -->
<section id="errors">
<h2>Error codes</h2>
<table>
<tr><th>Code</th><th>Meaning</th><th>What to do</th></tr>
<tr><td class="num">400</td><td>Invalid parameters (e.g. pixel cap exceeded)</td><td>Fix the request</td></tr>
<tr><td class="num">401</td><td>Missing or wrong bearer token</td><td>Set the <code class="inl">Authorization</code> header</td></tr>
<tr><td class="num">404</td><td>Unknown job id</td><td>Check the id</td></tr>
<tr><td class="num">409</td><td>Job not finished (result fetch) or already rendering (cancel)</td><td>Poll until done; cancel only while queued</td></tr>
<tr><td class="num">410</td><td>Result expired (24 h)</td><td>Generate again</td></tr>
<tr><td class="num">422</td><td>Parameter failed validation</td><td>The response names the field</td></tr>
<tr><td class="num">500</td><td>Render failed server-side (rare; e.g. repeated OOM)</td><td>Retry once, then report</td></tr>
<tr><td class="num">503</td><td><code class="inl">fast</code> requested but the fast lane isn't loaded</td><td>Drop <code class="inl">fast</code> or wait for the operator</td></tr>
</table>
<p>A transport drop (timeout, connection reset) is never a verdict — the job stays in
the queue. Poll again with the same id.</p>
</section>

<!-- ============================================================ limits -->
<section id="limits">
<h2>Limits &amp; configuration</h2>
<h3>Per-request limits</h3>
<table>
<tr><th>Limit</th><th>Value</th><th>Notes</th></tr>
<tr><td>Resolution</td><td class="num">64–2048 px per side, ≤ 2048×2048 total-capped</td><td>Model native is 2K; multiples of 32</td></tr>
<tr><td>Steps</td><td class="num">1–50</td><td>Linear cost; fast lane pins 8</td></tr>
<tr><td>Result retention</td><td class="num">24 h</td><td>PNGs are deleted after this</td></tr>
<tr><td>Sync wait (OpenAI route)</td><td class="num">120 s</td><td>Then returns the 202 envelope</td></tr>
<tr><td>Concurrent renders</td><td class="num">3</td><td>More requests queue with positions</td></tr>
</table>
<h3>Operator environment variables</h3>
<table>
<tr><th>Variable</th><th>Default</th><th>Purpose</th></tr>
<tr><td><code>QI21_API_KEY</code></td><td class="default">required</td><td>Bearer token for all authenticated routes</td></tr>
<tr><td><code>QI21_PORT</code></td><td class="default">8080</td><td>Listen port</td></tr>
<tr><td><code>QI21_WORKERS</code></td><td class="default">3</td><td>Replicas — 3 is the proven maximum for full-weight 2K</td></tr>
<tr><td><code>QI21_MAX_STEPS</code></td><td class="default">50</td><td>Server-side steps ceiling</td></tr>
<tr><td><code>QI21_MAX_PIXELS</code></td><td class="default">4194304</td><td>Pixel cap (2048×2048)</td></tr>
<tr><td><code>QI21_SYNC_WAIT_S</code></td><td class="default">120</td><td>OpenAI-route wait budget</td></tr>
<tr><td><code>QI21_RENDER_LEASE_S</code></td><td class="default">900</td><td>Crash-lease before a rendering job is requeued</td></tr>
<tr><td><code>QI21_RESULT_TTL_S</code></td><td class="default">86400</td><td>Result retention</td></tr>
</table>
</section>

<!-- ============================================================ performance -->
<section id="performance">
<h2>Measured performance</h2>
<p>Real numbers from this deployment (H200, full BF16, warm). Per-image cost at
$3.82/h is about <b>$0.006 at 1024²/40 steps</b>.</p>
<table>
<tr><th>Setting</th><th>Render time</th><th>Rate</th></tr>
<tr><td class="num">512² · 20 steps</td><td class="num">~0.9 s</td><td class="num">0.05 s/step</td></tr>
<tr><td class="num">1024² · 20 steps</td><td class="num">3.0 s</td><td class="num">0.15 s/step</td></tr>
<tr><td class="num">1024² · 40 steps</td><td class="num">5.9 s</td><td class="num">0.15 s/step</td></tr>
<tr><td class="num">2048² · 20 steps</td><td class="num">15.8 s</td><td class="num">0.79 s/step</td></tr>
<tr><td class="num">2048² · 50 steps (max)</td><td class="num">~42 s</td><td class="num">0.85 s/step</td></tr>
<tr><td class="num">1024² · 8 steps (fast lane)</td><td class="num">~1.5 s</td><td class="num">0.19 s/step</td></tr>
</table>
<p>Under burst the workers share the GPU, so concurrent renders each run slower
while total throughput rises — a measured 5-way burst of 512²/20 sustained
<b>51 images/min</b> with every request served.</p>
</section>

<footer>
Qwen-Image-2.1 weights are provided under the Qwen Research License
(non-commercial research and evaluation). This service is operated for
non-commercial community testing. Base image: full BF16, unquantized,
three replicas on one NVIDIA H200.
</footer>

</main>
</div>

<script>
"use strict";
// ---- copy buttons ----
document.querySelectorAll(".copy").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var text = btn.getAttribute("data-copy") || "";
    var done = function () {
      var prev = btn.innerHTML;
      btn.classList.add("ok");
      btn.textContent = "copied";
      setTimeout(function () { btn.classList.remove("ok"); btn.innerHTML = prev; }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, done);
    } else {
      var ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); ta.remove(); done();
    }
  });
});

// ---- mobile slide-in drawer ----
var ham = document.querySelector(".hamburger");
var side = document.querySelector("nav.side");
var backdrop = document.querySelector(".backdrop");
function setDrawer(open) {
  side.classList.toggle("open", open);
  backdrop.classList.toggle("show", open);
  ham.setAttribute("aria-expanded", open ? "true" : "false");
  document.documentElement.style.overflow = open ? "hidden" : "";
}
ham.addEventListener("click", function () { setDrawer(!side.classList.contains("open")); });
backdrop.addEventListener("click", function () { setDrawer(false); });
document.addEventListener("keydown", function (e) {
  if (e.key === "Escape" && side.classList.contains("open")) { setDrawer(false); ham.focus(); }
});
// picking a destination closes the drawer and jumps straight there
links.forEach(function (a) {
  a.addEventListener("click", function () {
    if (window.matchMedia("(max-width: 960px)").matches) setDrawer(false);
  });
});

// ---- scroll-spy for the sidebar ----
var links = Array.prototype.slice.call(document.querySelectorAll("nav.side a"));
var sections = links.map(function (a) {
  return document.getElementById(a.getAttribute("href").slice(1));
}).filter(Boolean);

function spy() {
  var fromTop = window.scrollY + 90;
  var active = sections[0];
  sections.forEach(function (s) { if (s.offsetTop <= fromTop) active = s; });
  links.forEach(function (a) {
    a.classList.toggle("active", a.getAttribute("href") === "#" + active.id);
  });
}
window.addEventListener("scroll", spy, { passive: true });
spy();
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
@app.get("/")
async def docs():
    """The API reference. The service has no web generator by design — this
    page IS the human interface; everything else is JSON."""
    return HTMLResponse(DOCS_HTML)


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
