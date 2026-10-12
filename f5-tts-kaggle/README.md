# F5-TTS on Kaggle T4 ×2: SOTA Zero-Shot Voice Cloning & Speech API

Deploy **F5-TTS v1 Base** on Kaggle's free dual NVIDIA T4 GPUs. Produces natural, human-like voice synthesis with zero-shot voice cloning from just **5 to 10 seconds** of reference audio (no 1–5 minute fine-tuning dataset required).

Features an interactive **Web Voice Studio UI** for mobile and desktop, an **OpenAI-compatible `/v1/audio/speech` endpoint**, pre-bundled studio voice presets, and an automated Cloudflare quick tunnel.

---

## Key Highlights

- **5–10s Zero-Shot Voice Cloning**: Flow Matching Diffusion Transformer captures exact speaker timbre, pitch, inflection, and breathing pauses from a short audio clip.
- **Ultra-Lightweight Footprint**: Base model weighs only **1.5 GB**, occupying **~2.8 GB VRAM** on GPU 0. Leaves ~13 GB VRAM headroom.
- **Fast Synthesis**: Real-Time Factor (RTF) of ~0.22x–0.28x on T4 (~4x faster than real-time speech).
- **Interactive Web Studio**: Built-in responsive HTML5 studio accessible via mobile browser (Kiwi, Chrome) and desktop (Firefox, Linux). Allows instant in-browser recording/uploading and playback.
- **OpenAI `/v1/audio/speech` Compatible**: Drop-in replacement for OpenAI TTS in third-party readers, mobile apps, and scripts.

---

## Memory & Performance Breakdown (Kaggle T4 ×2)

| Component | Footprint | Notes |
|---|---|---|
| **F5-TTS v1 Base Weights** | **1.48 GB** | 1024-dim DiT model with ConvNeXt V2 text encoder |
| **Vocos Mel-24kHz Vocoder** | **~50 MB** | High-fidelity neural vocoder generating 24kHz audio |
| **Diffusion KV & Activations** | **~1.2 GB** | Transient working memory during flow matching ODE solve |
| **Total VRAM on GPU 0** | **~2.8 GB** (out of 15.8 GB) | **100% on GPU 0 — Zero PCIe cross-GPU latency!** |
| **GPU 1** | **0 GB** (idle) | Completely free for other tasks |
| **Real-Time Factor (RTF)** | **~0.25x** | 10 seconds of speech generated in ~2.5 seconds |

---

## Notebook Cells (Run in Order)

In a Kaggle Notebook with **GPU T4 ×2** and **Internet ON**:

### 1. Setup Dependencies
```python
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/f5-tts-kaggle/cell_00_deps.sh | sh
```
*Installs system audio codecs (`ffmpeg`, `libsndfile1`), build toolchain, and Hugging Face transfer packages.*  
*Success sentinel: `DEPS_OK`*

---

### 2. Pull / Synchronize Kit
```python
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/f5-tts-kaggle/cell_00b_pull.sh | sh
```
*Clones or fast-forwards the repository inside `/kaggle/working/kaggle-tpu-lab`.*  
*Success sentinel: `PULL_OK`*

---

### 3. Hardware & GPU Report
```python
!sh /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/cell_01_report.sh
```
*Inspects both T4 GPUs, driver, CUDA runtime, and available scratch space.*  
*Success sentinel: `REPORT_OK`*

---

### 4. Setup Python Environment & Cloudflare Tunnel
```python
!sh /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/cell_02_setup.sh
```
*Installs `f5-tts`, `fastapi`, `uvicorn`, `gradio`, and fetches `cloudflared`.*  
*Success sentinel: `SETUP_OK`*

---

### 5. Download Model Weights & Studio Reference Presets
```python
!python3 /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/cell_03_download.py
```
*Downloads F5-TTS v1 weights, Vocos vocoder, and bundles 4 studio voice clips into `/kaggle/tmp/f5tts/voices/`.*  
*Success sentinel: `DOWNLOAD_OK`*

*(Optional: check download status with `!sh /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/cell_03b_status.sh`)*

---

### 6. Smoke Test & RTF Measurement
```python
!python3 /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/cell_04_smoke.py
```
*Runs a test zero-shot generation on CUDA:0, measures latency, audio length, RTF, and validates generated WAV.*  
*Success sentinel: `SMOKE_OK`*

---

### 7. Launch Serving Stack & Cloudflare Quick Tunnel
```python
!sh /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/cell_05_serve.sh
```
*Launches the FastAPI server with concurrency locking, starts the Cloudflare tunnel, and prints the live public URL.*  
*Output sentinel: `F5TTS_ENDPOINT_OK`*

---

### 8. Automated Verification Client (Proof)
```python
!python3 /kaggle/working/kaggle-tpu-lab/f5-tts-kaggle/cell_05b_proof.py
```
*Tests `/health`, queries `/v1/audio/voices`, calls `/v1/audio/speech`, and verifies valid WAV RIFF headers.*  
*Output sentinel: `PROOF_OK`*

---

## How to Use

### A. Web Voice Studio (Mobile & Desktop)
Open the generated `https://*.trycloudflare.com` URL in any browser (Firefox on Linux, Kiwi Browser / Chrome on Android):
1. **Interactive Audio Visualizer**: Watch real-time frequency waves on the canvas player as your audio synthesizes and plays back.
2. **Preset Voice Cards**: Switch effortlessly between:
   - **Nature / Documentary** (`nature_en`) — Warm cinematic English narration.
   - **Conversational** (`mandarin_zh`) — Fluent native Mandarin Chinese with natural inflection.
   - **Storyteller** (`storyteller`) — Expressive pacing and character cadence.
   - **Narrator** (`narrator`) — Crisp, authoritative audiobook voice.
3. **Instant Microphone Voice Cloning**:
   - Click **Record from Mic** directly on your phone or laptop.
   - Speak for 5 to 10 seconds, then click Stop.
   - Type the words you spoke, enter your target text, and F5-TTS will immediately replicate your voice!
4. **File Upload Option**: Or upload any pre-recorded WAV, MP3, or M4A clip.
5. **Flow Matching Controls**: Tune ODE steps (16 to 48 steps) and cadence speed (0.5x to 1.8x).
6. **1-Click Download**: Save the generated 24kHz studio-quality WAV directly to your device.

### B. OpenAI-Compatible Speech API (`/v1/audio/speech`)
Send standard POST requests compatible with OpenAI's audio API:

```bash
curl -X POST https://<TUNNEL-URL>/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "f5-tts",
    "input": "This is synthetic human speech generated with flow matching.",
    "voice": "nature_en",
    "speed": 1.0
  }' \
  --output speech.wav
```

### C. Python OpenAI Client Example
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<TUNNEL-URL>/v1",
    api_key="none", # or your F5_API_KEY
)

response = client.audio.speech.create(
    model="f5-tts",
    voice="nature_en",
    input="F5-TTS produces natural speech cadence and emotional prosody.",
)

response.stream_to_file("output.wav")
```

---

## Safety & Watchdog
- **Ceiling**: A background watchdog stops the server and tunnel after **240 minutes** (4 hours) to prevent running down session limits.
- **Manual Stop**: Run `!pkill -f server.py; pkill -f cloudflared` in any notebook cell to immediately terminate the endpoint.
