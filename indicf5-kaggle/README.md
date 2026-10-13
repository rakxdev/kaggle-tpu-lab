# IndicF5 Hinglish TTS — Kaggle Dual T4 Kit

> **Zero-Shot Hinglish Voice Cloning with Shubh Persona on Kaggle 2x NVIDIA T4**  
> Powered by **AI4Bharat IndicF5 (330M Diffusion Transformer)**, **Mode A Duration Patch**, and **IndicXlit Unified Devanagari Normalizer**.

---

## 1. Overview & Key Capabilities

- **Natural Warm Storyteller Voice ("Shubh")**: Bundles a reference audio clip (`shubh_ref.wav`, 5.46s @ 24 kHz Mono) derived from the winning Bulbul v3 benchmark for authentic Hindi & Hinglish storytelling and YouTube Shorts/Reels voiceovers.
- **Mode A Duration Bug Resolved**: Fixes upstream IndicF5's UTF-8 byte counting flaw (`len(text.encode("utf-8"))`) by counting non-whitespace characters, eliminating truncation and hurried audio when mixing Devanagari and Latin scripts.
- **IndicXlit Unified Devanagari Preprocessing**: Converts Romanized Hinglish words (`office` → `ऑफिस`, `meeting` → `मीटिंग`, `presentation` → `प्रेज़ेंटेशन`) so the neural vocoder and diffusion model never encounter untrained ASCII character embeddings.
- **Python 3.13 / Kaggle Conflict Isolation**: Uses `uv` to provision a standalone CPython 3.11 environment in `/kaggle/tmp/indicf5/venv`, bypassing Kaggle's incompatible system packages completely.
- **GPU Turing T4 Footprint**: Requires only **~3.5 GiB VRAM** on GPU 0, leaving **>12 GiB VRAM headroom**.
- **Interactive Web Studio + OpenAI API**: Features a DAW-style dark Web Studio UI with HTML5 Canvas audio spectrum visualizer and standard OpenAI-compatible `/v1/audio/speech` endpoint.

---

## 2. Step-by-Step Kaggle Notebook Cells

Copy and paste these commands into individual cells in your Kaggle notebook (configured with **GPU: 2x T4**, **Internet: ON**):

### Cell 0 — Install System Audio Codecs & uv
```bash
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/indicf5-kaggle/cell_00_deps.sh | bash
```
*Expected sentinel*: `DEPS_OK`

---

### Cell 0b — Clone or Pull Latest Repository
```bash
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/indicf5-kaggle/cell_00b_pull.sh | bash
```
*Expected sentinel*: `PULL_OK`

---

### Cell 1 — Hardware Inspection Report
```bash
!bash /kaggle/working/kaggle-tpu-lab/indicf5-kaggle/cell_01_report.sh
```
*Expected sentinel*: `REPORT_OK` (verifies Dual T4 GPUs and memory)

---

### Cell 2 — Provision Isolated Python 3.11 & Build Dependencies
```bash
!bash /kaggle/working/kaggle-tpu-lab/indicf5-kaggle/cell_02_setup.sh
```
*Expected sentinel*: `SETUP_OK` (provisions uv Python 3.11, fairseq, ai4bharat-transliteration, IndicXlit weights, and applies duration patch)

---

### Cell 3 — Download Model Checkpoints & Stage Shubh Voice
```bash
!python3 /kaggle/working/kaggle-tpu-lab/indicf5-kaggle/cell_03_download.py
```
*Expected sentinel*: `DOWNLOAD_OK` (authenticates via encrypted token resolver and downloads `ai4bharat/IndicF5` ~1.4 GB weights)

*(Optional progress monitor: `!bash /kaggle/working/kaggle-tpu-lab/indicf5-kaggle/cell_03b_status.sh`)*

---

### Cell 4 — Zero-Shot Voice Cloning Smoke Test
```bash
!python3 /kaggle/working/kaggle-tpu-lab/indicf5-kaggle/cell_04_smoke.py
```
*Expected sentinel*: `SMOKE_OK` (synthesizes Devanagari and Hinglish test sentences, reports RTF and verifies VRAM footprint ~3.5 GiB)

---

### Cell 5 — Launch Web Studio Server & Cloudflare Public Tunnel
```bash
!bash /kaggle/working/kaggle-tpu-lab/indicf5-kaggle/cell_05_serve.sh
```
*Expected sentinel*: `INDICF5_ENDPOINT_OK` (prints public HTTPS URL `https://*.trycloudflare.com`)

---

### Cell 5b — Automated Live Endpoint Verification (Optional)
```bash
!python3 /kaggle/working/kaggle-tpu-lab/indicf5-kaggle/cell_05b_proof.py
```
*Expected sentinel*: `PROOF_OK` (verifies `/health`, `/v1/audio/voices`, and generates test audio)

---

## 3. OpenAI-Compatible API Usage

Once the server is running, any client or Python script can generate cloned speech:

### cURL Example
```bash
curl -X POST https://<TUNNEL_URL>/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "indicf5",
    "input": "कल office में new project launch की meeting है, सब smoothly हो जाएगा।",
    "voice": "shubh",
    "speed": 1.0,
    "normalize": true
  }' \
  --output speech.wav
```

### Python (OpenAI SDK) Example
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<TUNNEL_URL>/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="indicf5",
    voice="shubh",
    input="बारिश की एक ठंडी शाम थी। शहर की एक छोटी सी चाय की दुकान पर मैं अपनी चाय का इंतज़ार कर रहा था।"
)

response.stream_to_file("cloned_shubh.wav")
```

---

## 4. Hardware & Memory Footprint

| Metric | Measured on Kaggle Turing T4 |
|---|---|
| **Base Model VRAM** | ~2.65 GiB |
| **Peak Inference VRAM** | ~3.48 GiB |
| **GPU Headroom** | **>12.3 GiB Free** (GPU 1 completely idle) |
| **Output Audio Format** | 24,000 Hz, 16-bit Mono WAV |
| **Real-Time Factor (RTF)** | ~0.70x – 0.95x (faster than real-time) |
