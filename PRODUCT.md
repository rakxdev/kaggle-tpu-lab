# PRODUCT.md — Qwen-Image-2.1 Community API

Source: user brief + AskUserQuestion answers, 2026-10-06. Single-surface project.

## What this is

A community image-generation API serving the full-weight Qwen-Image-2.1 model
(33.1 GB BF16, unquantized) from one NVIDIA H200 on Lightning AI. Three worker
processes, a never-failing FIFO job queue (SQLite WAL), VRAM-aware scheduling,
and an opt-in Turbo8 distilled fast lane.

## Who it's for

The owner's community members: developers who integrate over HTTP. No web
generator — the service is API-only by explicit decision. The docs page at `/`
is the product surface for humans; everything else is JSON.

## Success looks like

A developer lands on `/`, understands the whole API within a minute (auth,
submit, poll, fetch result), copies a working curl/Python example, and never
experiences a rejected request — the queue holds their place instead.

## Surface

One page: the API reference at `/`. Mode: **Read** (structure for
comprehension). Dark theme, modern industry docs conventions (Stripe/Mintlify
school): sticky sidebar, method badges, copyable examples, params tables.

## Constraints

- Self-contained single HTML file, zero external dependencies (behind an ngrok
  tunnel; CDN calls must never be load-bearing).
- Accuracy over polish: every endpoint, default, cap, and error code must match
  `qi21_server.py` exactly.
- One factual license line in the footer (Qwen Research, non-commercial).
  No repeated license discussion anywhere else.
