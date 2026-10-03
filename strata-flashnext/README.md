# Strata on Kaggle 2x T4 — Qwen3.8-Flash-Next Coder (IQ1_M)

Run the 125B Flash-Next **Coder** (256 of 512 experts, code-specialized,
authors report 91% SWE-bench Verified / 99% LiveCodeBench of the full model)
on Kaggle's free 2x T4 via [Strata](https://github.com/Niko1221/Strata) —
its streaming expert engine, ready-made sm_75 (Turing) binary, OpenAI **and**
Anthropic APIs on one port.

**Session: GPU T4x2, Internet ON.** Nothing else. Driver 580+ is Kaggle's.

## The cells (run in order, one at a time — each is small on purpose)

| Cell | File | Job | Time |
|---|---|---|---|
| 1 | `cell_01_report.sh` | machine report (2x T4, driver, RAM, disk, both pythons) | 5 s |
| 2 | `cell_02_clone.sh` | **clean-pull this kit** (hard reset + delete untracked, so every driver fix lands clean) + clone/update Strata + venv-capable python + pre-build Strata's `.venv` + arm the GPU heartbeat | 1–2 min |
| 3 | `cell_03_creds.py` | paste HF + ngrok tokens inline (go to /tmp only) + verify | 30 s |
| 4 | `cell_04_download.py` | Coder GGUF (58.4 GB) via hf_transfer to /kaggle/tmp — resumable, deps auto-installed | 15–25 min |
| 5 | `cell_05_setup.sh` (+ `cell_05b_status.sh` to poll) | setup: engine + MTP draft + prepare (uses the pre-built `.venv`) | 20–40 min |
| 6 | `cell_06_smoke.py` | local health + proof + first tok/s | 1 min |
| 7 | `cell_07_endpoint.sh` | **stable ngrok domain** + API key + keepalive watchdog | 1 min |

Run pattern from the notebook (repo already cloned by cell 2):

```python
!bash /kaggle/working/kaggle-tpu-lab/strata-flashnext/cell_01_report.sh
```
(.py cells: `!python3 /kaggle/working/kaggle-tpu-lab/strata-flashnext/cell_04_download.py`)

**Re-running after a driver fix:** just run cell 2 again — it hard-resets the
kit repo and deletes untracked files, so the pushed fixes replace everything
cleanly. Your Strata checkout, `.venv`, API key and logs are outside the repo
dir and survive untouched.

**The endpoint** is a static ngrok domain — same URL every session, both APIs
(OpenAI `/v1` + Anthropic `/v1/messages`, streaming + tools — the server speaks
both natively), API-key protected, and a keepalive watchdog that shuts the
server + tunnel down after 480 min so a forgotten session doesn't burn quota.
No ntfy relay in this lane: run the cells, paste me any error output.

## Facts this kit is built on (from Strata's source, checked 2026-10-03)

- Ready-made engine covers **RTX 20 / sm_75 since v0.1.27** (contributor-tested
  on an RTX 2070) — no 20–40 min compile expected; if setup offers to compile
  anyway, accept: Kaggle has CUDA + gcc.
- Coder = `--family coder --model IQ1_M`, HF repo
  `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, download 58.4 GB,
  `ram_gb: 32` — Kaggle has ~31 GB, and Strata's low-RAM mode (GGUF-in-place)
  covers the gap when the GPU is big; ours is 32 GB VRAM total.
- `--gguf-dir` accepts GGUFs you already have — that is why cell 4 pre-downloads
  with your HF token at hf_transfer speed instead of letting setup download.
- The server speaks OpenAI `/v1/chat/completions` AND Anthropic
  `/v1/messages` (streaming + tools) on :8080; unknown Host names get 403 by
  design, so cell 7 writes `api_key` + `allowed_hosts` and exports
  `STRATA_ALLOWED_HOSTS` for the tunnel hostname.
- Server = constant GPU activity: no idle-stop risk while serving.

## Honest expectations

- Speed: their tables are 12–24 GB GeForce cards with 6-core desktop CPUs;
  Kaggle's 4 vCPUs are the weak link for the CPU-side expert streaming.
  Hope: 30–55 tok/s. Anything above ~20 tok/s makes it usable for coding agents.
- RAM 31 GB vs the Coder's 32 GB target — the low-RAM mode is designed for
  exactly this; if setup refuses, re-run cell 5 adding `--yes` (already there)
  and check the log for the tier it chose.
- Fresh project (v0.1.x): first friction goes into a fix commit, not a dead end.

## If something fails

- Cell 5 log: `/kaggle/working/strata_setup.log` (tail with cell_05b).
- Server log: `/kaggle/working/strata_serve.log`; tunnel log:
  `/kaggle/working/tunnel.log` (re-run cell 7 for a fresh URL).
- A fresh session needs cells 1–5 re-run (the 58 GB GGUF re-downloads — the
  dataset-packing optimization comes after the route is proven once).
