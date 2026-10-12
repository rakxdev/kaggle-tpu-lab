# IndexTTS-2.5 on Kaggle T4 ×2: Industrial 1.5B Emotion Speech Synthesis

Deploy **IndexTTS-2.5** (Bilibili's state-of-the-art 1.5B parameter TTS model, arXiv `2601.03888`) on Kaggle's dual NVIDIA T4 GPUs. Features **Disentangled Timbre & Emotion Control**, allowing you to clone any speaker's voice from a 5–10 second clip while applying rich theatrical emotions (joy, rage, grief, terror, whispering, laughter) via an 8-axis emotion board, natural language prompts, or separate emotional audio clips.

---

## Why IndexTTS-2.5 Over Lightweight Models

| Capability | Lightweight (F5-TTS, ~386M) | **IndexTTS-2.5 (~1.5B Architecture)** |
|---|---|---|
| **Emotional Expressiveness** | Fixed to reference clip tone (often flat) | **Full Theatrical Emotion (Rage, Tears, Laughter, Whispering, Joy)** |
| **Emotion Control Modes** | None | **8-Axis Sliders, Natural Language Text, or Separate Emotional Audio** |
| **Timbre-Emotion Disentanglement** | ❌ No | **✅ Yes (Your voice + Emotional actor's performance)** |
| **Model Capacity** | ~386 Million parameters | **~1.5 Billion parameters (Studio Grade)** |
| **VRAM on T4 (FP16)** | ~2.8 GB | **~6.8 – 7.5 GB on GPU 0** (Leaves ~8.3 GB headroom) |
| **Language Support** | English & Chinese | **English, Chinese, Japanese, Spanish, Arabic** |
| **Speed Control** | Heuristic duration stretch | **Precise `duration_factor` duration conditioning (0.5x–2.0x)** |

---

## Memory & Performance Breakdown (Kaggle T4 ×2)

| Component | Footprint | Details |
|---|---|---|
| **IndexTTS-2.5 GPT Weights (`gpt.pth`)** | **~2.8 GB** | Autoregressive acoustic duration & semantic transformer |
| **S2Mel Flow Decoder (`s2mel.pth`)** | **~1.6 GB** | Mel-spectrogram synthesis diffusion head |
| **Audio Codec Vocoder (`codec.pth`)** | **~0.4 GB** | 24kHz neural audio decoder |
| **Emotion Conditioning (`feat1.pt`, `feat2.pt`)** | **~0.3 GB** | 8-axis emotion latent projection space |
| **Total VRAM on GPU 0** | **~7.2 GB** / 15.8 GB | **Runs 100% on GPU 0 in FP16 with ~8.6 GB headroom** |
| **GPU 1** | **0 GB** (idle) | Free for second concurrent worker |

---

## Notebook Cells (Run in Order)

In a Kaggle Notebook with **GPU T4 ×2** and **Internet ON**:

### 1. Setup Audio Codecs & Toolchain
```python
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/indextts-kaggle/cell_00_deps.sh | sh
```
*Installs system audio codecs (`ffmpeg`, `libsndfile1`, `sox`), build toolchain, and `uv`.*  
*Success sentinel: `DEPS_OK`*

---

### 2. Pull / Synchronize Kit
```python
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/indextts-kaggle/cell_00b_pull.sh | sh
```
*Pulls the latest repository changes into `/kaggle/working/kaggle-tpu-lab`.*  
*Success sentinel: `PULL_OK`*

---

### 3. Hardware & GPU Report
```python
!sh /kaggle/working/kaggle-tpu-lab/indextts-kaggle/cell_01_report.sh
```
*Queries both T4 GPUs, driver, CUDA runtime, and available scratch space.*  
*Success sentinel: `REPORT_OK`*

---

### 4. Setup IndexTTS Repository & Python Stack
```python
!sh /kaggle/working/kaggle-tpu-lab/indextts-kaggle/cell_02_setup.sh
```
*Clones the official `index-tts` engine, installs speech dependencies, and fetches `cloudflared`.*  
*Success sentinel: `SETUP_OK`*

---

### 5. Download Model Weights (~5 GB) & Example Clips
```python
!python3 /kaggle/working/kaggle-tpu-lab/indextts-kaggle/cell_03_download.py
```
*Downloads `IndexTeam/IndexTTS-2.5` model weights from Hugging Face into `/kaggle/tmp/indextts/checkpoints`.*  
*Success sentinel: `DOWNLOAD_OK`*

*(Optional status monitor: `!sh /kaggle/working/kaggle-tpu-lab/indextts-kaggle/cell_03b_status.sh`)*

---

### 6. Smoke Test (Voice Cloning + Emotion Vector)
```python
!python3 /kaggle/working/kaggle-tpu-lab/indextts-kaggle/cell_04_smoke.py
```
*Tests zero-shot voice cloning and applies an 8-axis emotion vector (`Happy 0.9 + Surprised 0.5`) on CUDA:0 in FP16.*  
*Success sentinel: `SMOKE_OK`*

---

### 7. Launch Master Emotion Studio & Cloudflare Quick Tunnel
```python
!sh /kaggle/working/kaggle-tpu-lab/indextts-kaggle/cell_05_serve.sh
```
*Starts the FastAPI server with concurrency locking, launches the Cloudflare quick tunnel, arms the 4-hour watchdog, and outputs the public URL.*  
*Output sentinel: `INDEXTTS_ENDPOINT_OK`*

---

### 8. Automated Verification Client (Proof)
```python
!python3 /kaggle/working/kaggle-tpu-lab/indextts-kaggle/cell_05b_proof.py
```
*Tests `/health`, calls `/v1/audio/speech` with emotion vectors, and validates 24kHz audio headers.*  
*Output sentinel: `PROOF_OK`*

---

## Emotion Control Guide

IndexTTS-2.5 provides **3 distinct emotion control modalities**:

### 1. The 8-Axis Emotion Vector (`emo_vector`)
Pass an 8-float array representing the intensity (`0.0` to `1.0`) of each emotion in order:
`[Happy, Angry, Sad, Afraid, Disgusted, Melancholic, Surprised, Calm]`

*Example: Pure Dramatic Rage:*
```json
{
  "emo_vector": [0.0, 0.95, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
  "emo_alpha": 0.9
}
```

*Example: Jubilant Celebration:*
```json
{
  "emo_vector": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
  "emo_alpha": 0.85
}
```

### 2. Dual-Audio Reference Transfer (`emo_audio`)
You can provide **two separate audio clips**:
1. `spk_audio`: 5-second clip of **your voice** (supplies the vocal timbre and identity).
2. `emo_audio`: Clip of an **actor screaming, crying, or laughing** (supplies the dramatic energy).
IndexTTS-2.5 extracts the emotion from clip 2 and applies it to your cloned voice in clip 1!

### 3. Emotion Intensity (`emo_alpha`)
Adjusts how forcefully the emotion shapes the pronunciation.
- `0.2 – 0.5`: Subtle, natural conversational warmth.
- `0.6 – 0.85`: Standard expressive performance (recommended).
- `0.9 – 1.0`: Extreme theatrical intensity.

---

## OpenAI Speech API Specification (`/v1/audio/speech`)

```bash
curl -X POST https://<TUNNEL-URL>/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "indextts-2.5",
    "input": "I cannot believe this! We actually won the entire championship!",
    "lang": "EN",
    "emo_vector": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    "emo_alpha": 0.85
  }' \
  --output expressive.wav
```

---

## Safety Watchdog
- **Ceiling**: A background watchdog automatically terminates the server and tunnel after **240 minutes** (4 hours) to prevent running down session limits.
- **Manual Stop**: Run `!pkill -f server.py; pkill -f cloudflared` in any notebook cell.
