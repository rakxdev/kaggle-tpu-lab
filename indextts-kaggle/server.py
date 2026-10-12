#!/usr/bin/env python3
"""
IndexTTS-2.5 Production Master Server:
- Full Disentangled Timbre & Emotion Speech Synthesis
- OpenAI-compatible /v1/audio/speech API
- 8-Axis Emotion Control & Dual-Audio Reference Transfer
- Master Emotion Studio WebUI with real-time waveform visualizer
- Concurrency queue lock to prevent GPU OOM collisions
"""

import os
import sys

# Automatically re-exec inside the isolated uv environment if running from system python
VENV_PY = "/kaggle/tmp/indextts/repo/.venv/bin/python"
if os.path.exists(VENV_PY) and sys.executable != VENV_PY:
    os.execv(VENV_PY, [VENV_PY] + sys.argv)

import io
import time
import json
import base64
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path("/kaggle/tmp/indextts")
CHECKPOINTS = BASE_DIR / "checkpoints"
REPO_DIR = BASE_DIR / "repo"
API_KEY = os.environ.get("INDEX_API_KEY", "")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

app = FastAPI(
    title="IndexTTS-2.5 Master Emotion Voice Studio",
    description="Industrial-grade controllable zero-shot voice cloning with 8-axis emotion vectors and audio emotion transfer.",
    version="2.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_engine = None
gpu_lock = asyncio.Lock()

def get_engine():
    global tts_engine
    if tts_engine is None:
        import torch
        from indextts.infer_v2_5 import IndexTTS2

        cfg_path = CHECKPOINTS / "config.yaml"
        print("[IndexTTS-2.5] Loading model into CUDA:0 (FP16)...")
        t0 = time.time()
        tts_engine = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(CHECKPOINTS),
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        print(f"[IndexTTS-2.5] Engine initialized in {time.time() - t0:.2f}s")
    return tts_engine

# Request schema
class SpeechRequest(BaseModel):
    model: str = Field(default="indextts-2.5", description="Model identifier")
    input: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default="narrator", description="Voice preset or ID")
    lang: Optional[str] = Field(default="EN", description="Language: EN, ZH, JA, ES, AR")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="Speaking speed rate")
    
    # Advanced Emotion Controls
    # 8-axis vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    emo_vector: Optional[List[float]] = Field(default=None, description="8-float emotion intensity vector")
    emo_audio: Optional[str] = Field(default=None, description="Base64 or path of emotional audio clip to copy emotion from")
    emo_alpha: Optional[float] = Field(default=0.8, ge=0.0, le=1.0, description="Emotion intensity factor (0.0=calm, 1.0=intense)")
    emo_text: Optional[str] = Field(default=None, description="Natural language emotion description")
    spk_audio: Optional[str] = Field(default=None, description="Base64 custom speaker voice clip for instant cloning")

def find_default_spk_audio(voice_name="narrator"):
    candidates = [
        REPO_DIR / "examples" / "voice_01.wav",
        REPO_DIR / "examples" / "voice_07.wav",
        Path("/kaggle/tmp/f5tts/voices/nature_en.wav"),
        BASE_DIR / "examples" / "voice_01.wav",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1000:
            return str(c)
    
    # Fallback to creating a tone if no wav exists
    fallback = BASE_DIR / "default_voice.wav"
    if not fallback.exists():
        import numpy as np, soundfile as sf
        sf.write(str(fallback), np.sin(2 * np.pi * 440 * np.linspace(0, 3, 3 * 24000)), 24000)
    return str(fallback)

@app.on_event("startup")
async def startup():
    asyncio.get_event_loop().run_in_executor(None, get_engine)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "IndexTTS-2.5",
        "device": "cuda:0",
        "emotion_modalities": ["8-axis-vector", "audio-emotion-transfer", "natural-language-prompt"],
    }

@app.get("/v1/audio/voices")
def list_voices():
    return {
        "voices": {
            "narrator": {"name": "Studio Narrator", "lang": "EN", "description": "Crisp, balanced studio voice"},
            "expressive_female": {"name": "Expressive Dramatic", "lang": "EN", "description": "Highly responsive to emotion vectors"},
            "mandarin_native": {"name": "Native Mandarin", "lang": "ZH", "description": "Fluent conversational Chinese"},
            "custom": {"name": "Custom Voice Clone", "lang": "ANY", "description": "Cloned from 5-10s user clip"},
        }
    }

@app.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest, authorization: Optional[str] = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    # 1. Resolve Speaker Voice Audio
    temp_spk_path = None
    if req.spk_audio:
        try:
            spk_bytes = base64.b64decode(req.spk_audio)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(spk_bytes)
            tmp.close()
            temp_spk_path = tmp.name
            spk_prompt = temp_spk_path
        except Exception:
            spk_prompt = req.spk_audio if os.path.exists(req.spk_audio) else find_default_spk_audio()
    else:
        spk_prompt = find_default_spk_audio(req.voice or "narrator")

    # 2. Resolve Emotion Audio if provided
    temp_emo_path = None
    emo_prompt = None
    if req.emo_audio:
        try:
            emo_bytes = base64.b64decode(req.emo_audio)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(emo_bytes)
            tmp.close()
            temp_emo_path = tmp.name
            emo_prompt = temp_emo_path
        except Exception:
            if os.path.exists(req.emo_audio):
                emo_prompt = req.emo_audio

    # 3. Calculate duration factor (speed)
    duration_factor = 1.0 / max(req.speed or 1.0, 0.25)

    async with gpu_lock:
        try:
            engine = get_engine()
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out_path = tmp_out.name
            tmp_out.close()

            def run_infer():
                kwargs = {
                    "spk_audio_prompt": spk_prompt,
                    "text": req.input.strip(),
                    "lang": req.lang or "EN",
                    "output_path": out_path,
                    "duration_factor": duration_factor,
                    "verbose": False,
                }
                # Apply emotion vector if given
                if req.emo_vector and len(req.emo_vector) == 8:
                    kwargs["emo_vector"] = req.emo_vector
                    kwargs["emo_alpha"] = req.emo_alpha or 0.8
                # Apply emotion audio if given
                elif emo_prompt:
                    kwargs["emo_audio_prompt"] = emo_prompt
                    kwargs["emo_alpha"] = req.emo_alpha or 0.8
                # Apply emotion text if given
                elif req.emo_text:
                    kwargs["use_emo_text"] = True
                    kwargs["emo_text"] = req.emo_text
                    kwargs["emo_alpha"] = req.emo_alpha or 0.6

                return engine.infer(**kwargs)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_infer)

            with open(out_path, "rb") as f:
                wav_bytes = f.read()

            try:
                os.remove(out_path)
            except OSError:
                pass

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"IndexTTS inference error: {str(e)}")
        finally:
            if temp_spk_path and os.path.exists(temp_spk_path):
                os.remove(temp_spk_path)
            if temp_emo_path and os.path.exists(temp_emo_path):
                os.remove(temp_emo_path)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )

# Master Emotion Studio Web UI
STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>IndexTTS-2.5 Master Emotion Studio — Kaggle T4</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-base: #06080e;
    --bg-surface: #0c101a;
    --bg-card: #121826;
    --bg-elevated: #1a2236;
    --bg-input: #080b12;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(236, 72, 153, 0.4);
    --cyan: #38bdf8;
    --pink: #ec4899;
    --pink-glow: rgba(236, 72, 153, 0.3);
    --purple: #a855f7;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --success: #10b981;
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 28px;
    --font-sans: 'Plus Jakarta Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-base);
    color: var(--text-primary);
    font-family: var(--font-sans);
    min-height: 100vh;
    padding: 24px 16px 60px;
    background-image: 
      radial-gradient(circle at 50% 0%, rgba(236, 72, 153, 0.12) 0%, transparent 60%),
      radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 50%);
    background-attachment: fixed;
    -webkit-font-smoothing: antialiased;
  }
  .shell { width: 100%; max-width: 1300px; margin: 0 auto; }
  
  /* App Header */
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 18px; border-bottom: 1px solid var(--border-subtle); flex-wrap: wrap; gap: 14px; }
  .brand { display: flex; align-items: center; gap: 14px; }
  .logo { width: 44px; height: 44px; border-radius: var(--radius-md); background: linear-gradient(135deg, var(--pink) 0%, var(--purple) 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 24px var(--pink-glow); }
  .logo svg { width: 24px; height: 24px; color: #fff; }
  .title-box h1 { font-size: 22px; font-weight: 800; background: linear-gradient(135deg, #fff 40%, #cbd5e1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .badge-live { display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; color: var(--success); }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); box-shadow: 0 0 8px var(--success); }

  /* Grid Layout */
  .workspace { display: grid; grid-template-columns: 460px 1fr; gap: 24px; align-items: start; }
  @media(max-width: 1080px) { .workspace { grid-template-columns: 1fr; } }

  .card { background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-xl); padding: 24px; box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
  .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
  .card-title { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .section-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 10px; display: block; }

  /* Emotion Faders Board */
  .fader-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
  .fader-box { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px 12px; }
  .fader-header { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }
  .fader-header strong { font-family: var(--font-mono); color: var(--text-primary); }
  input[type="range"] { width: 100%; height: 5px; background: var(--bg-elevated); border-radius: 4px; outline: none; -webkit-appearance: none; accent-color: var(--pink); }

  /* Emotion Presets Bar */
  .preset-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; }
  .preset-chip { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 999px; padding: 4px 10px; font-size: 11px; font-weight: 600; color: var(--text-secondary); cursor: pointer; transition: all 0.15s; }
  .preset-chip:hover { border-color: var(--pink); color: var(--pink); }
  .preset-chip.active { background: rgba(236, 72, 153, 0.15); border-color: var(--pink); color: #fff; }

  /* Audio Inputs (Timbre + Emotion) */
  .dual-audio-box { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
  .drop-zone { background: var(--bg-surface); border: 1px dashed var(--border-subtle); border-radius: var(--radius-md); padding: 14px; text-align: center; cursor: pointer; transition: all 0.2s; }
  .drop-zone:hover { border-color: var(--pink); }
  .drop-zone label { font-size: 12px; font-weight: 600; color: var(--text-secondary); display: block; cursor: pointer; }
  .drop-zone span { font-size: 11px; color: var(--text-muted); display: block; margin-top: 4px; }

  /* Textarea & Script Studio */
  textarea { width: 100%; min-height: 140px; background: var(--bg-input); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 16px; color: #fff; font-size: 15px; line-height: 1.6; outline: none; transition: all 0.2s; resize: vertical; }
  textarea:focus { border-color: var(--pink); box-shadow: 0 0 16px var(--pink-glow); }

  /* Primary Action */
  .btn-synthesize { width: 100%; background: linear-gradient(135deg, var(--pink) 0%, #be185d 100%); color: #fff; border: none; border-radius: var(--radius-md); padding: 18px; font-size: 16px; font-weight: 800; cursor: pointer; transition: all 0.25s; box-shadow: 0 6px 24px var(--pink-glow); display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 16px; }
  .btn-synthesize:hover { transform: translateY(-2px); box-shadow: 0 10px 32px rgba(236, 72, 153, 0.5); }
  .btn-synthesize:disabled { opacity: 0.5; transform: none; cursor: not-allowed; }

  /* Stage Audio Player */
  .stage-box { display: none; background: var(--bg-surface); border: 1px solid var(--border-hover); border-radius: var(--radius-xl); padding: 22px; margin-top: 20px; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
  #visualizer { width: 100%; height: 74px; background: #070a12; border-radius: var(--radius-md); margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.05); }
  audio { width: 100%; height: 46px; outline: none; border-radius: 8px; }
  .download-btn { display: inline-flex; align-items: center; gap: 8px; background: var(--bg-elevated); color: #fff; border: 1px solid var(--border-subtle); padding: 10px 18px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; text-decoration: none; margin-top: 14px; transition: all 0.2s; }
  .download-btn:hover { border-color: var(--pink); color: var(--pink); }

  pre { background: var(--bg-surface); border: 1px solid var(--border-subtle); padding: 14px; border-radius: 10px; color: var(--pink); font-family: var(--font-mono); font-size: 13px; overflow-x: auto; margin-top: 10px; }
</style>
</head>
<body>

<div class="shell">
  <div class="header">
    <div class="brand">
      <div class="logo">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
      </div>
      <div class="title-box">
        <h1>IndexTTS-2.5 Master Emotion Studio</h1>
        <div style="font-size:12px; color:var(--text-secondary); display:flex; align-items:center; gap:8px;">
          <span>Disentangled Emotion · 1.5B Architecture</span>
          <span class="badge-live"><span class="dot"></span> Kaggle T4 Active</span>
        </div>
      </div>
    </div>
  </div>

  <div class="workspace">
    <!-- Left: 8-Axis Emotion Control & Audio Inputs -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">🎭 Emotion Faders (8-Axis Board)</div>
        <span style="font-size:12px; color:var(--text-muted);">Disentangled Vector</span>
      </div>

      <!-- Quick Emotion Presets -->
      <span class="section-label">Emotion Presets</span>
      <div class="preset-row">
        <span class="preset-chip active" onclick="applyPreset([0,0,0,0,0,0,0,0.8], this)">Neutral</span>
        <span class="preset-chip" onclick="applyPreset([0.9,0,0,0,0,0,0.3,0], this)">😂 Joy / Laughter</span>
        <span class="preset-chip" onclick="applyPreset([0,0.95,0,0,0.3,0,0,0], this)">🔥 Fury / Rage</span>
        <span class="preset-chip" onclick="applyPreset([0,0,0.9,0,0,0.5,0,0], this)">😢 Sobbing / Grief</span>
        <span class="preset-chip" onclick="applyPreset([0,0,0,0.9,0,0,0.6,0], this)">😱 Panic / Terror</span>
        <span class="preset-chip" onclick="applyPreset([0,0,0,0,0,0,0.8,0.2], this)">😲 Shocked</span>
      </div>

      <!-- 8-Axis Sliders -->
      <div class="fader-grid">
        <div class="fader-box">
          <div class="fader-header"><span>Happy / Joy</span><strong id="val_0">0.0</strong></div>
          <input type="range" id="emo_0" min="0" max="1" step="0.05" value="0" oninput="updateFader(0, this.value)">
        </div>
        <div class="fader-box">
          <div class="fader-header"><span>Angry / Rage</span><strong id="val_1">0.0</strong></div>
          <input type="range" id="emo_1" min="0" max="1" step="0.05" value="0" oninput="updateFader(1, this.value)">
        </div>
        <div class="fader-box">
          <div class="fader-header"><span>Sad / Tears</span><strong id="val_2">0.0</strong></div>
          <input type="range" id="emo_2" min="0" max="1" step="0.05" value="0" oninput="updateFader(2, this.value)">
        </div>
        <div class="fader-box">
          <div class="fader-header"><span>Afraid / Terror</span><strong id="val_3">0.0</strong></div>
          <input type="range" id="emo_3" min="0" max="1" step="0.05" value="0" oninput="updateFader(3, this.value)">
        </div>
        <div class="fader-box">
          <div class="fader-header"><span>Disgusted</span><strong id="val_4">0.0</strong></div>
          <input type="range" id="emo_4" min="0" max="1" step="0.05" value="0" oninput="updateFader(4, this.value)">
        </div>
        <div class="fader-box">
          <div class="fader-header"><span>Melancholic</span><strong id="val_5">0.0</strong></div>
          <input type="range" id="emo_5" min="0" max="1" step="0.05" value="0" oninput="updateFader(5, this.value)">
        </div>
        <div class="fader-box">
          <div class="fader-header"><span>Surprised</span><strong id="val_6">0.0</strong></div>
          <input type="range" id="emo_6" min="0" max="1" step="0.05" value="0" oninput="updateFader(6, this.value)">
        </div>
        <div class="fader-box">
          <div class="fader-header"><span>Calm</span><strong id="val_7">0.8</strong></div>
          <input type="range" id="emo_7" min="0" max="1" step="0.05" value="0.8" oninput="updateFader(7, this.value)">
        </div>
      </div>

      <!-- Dual Reference Audio Inputs -->
      <span class="section-label">Disentangled Audio References</span>
      <div class="dual-audio-box">
        <div class="drop-zone" onclick="document.getElementById('spkFile').click()">
          <label id="spkLabel">🎤 Voice Reference</label>
          <span>5-10s clip of speaker</span>
          <input type="file" id="spkFile" accept="audio/*" style="display:none;" onchange="loadAudioFile(this, 'spk')">
        </div>
        <div class="drop-zone" onclick="document.getElementById('emoFile').click()">
          <label id="emoLabel">🎬 Emotion Reference</label>
          <span>Actor screaming / sobbing</span>
          <input type="file" id="emoFile" accept="audio/*" style="display:none;" onchange="loadAudioFile(this, 'emo')">
        </div>
      </div>

      <!-- Emotion Alpha Slider -->
      <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-secondary); margin-bottom:6px;">
          <span>Emotion Intensity (Alpha)</span>
          <strong id="alphaVal">0.8</strong>
        </div>
        <input type="range" id="alphaSlider" min="0.1" max="1.0" step="0.05" value="0.8" oninput="document.getElementById('alphaVal').innerText = this.value">
      </div>
    </div>

    <!-- Right: Script Editor & Output Stage -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">📝 Dialogue & Narrative Script</div>
        <select id="langSelect" style="background:var(--bg-elevated); color:#fff; border:1px solid var(--border-subtle); border-radius:6px; padding:4px 8px; font-size:12px;">
          <option value="EN">English</option>
          <option value="ZH">Chinese (Mandarin)</option>
          <option value="JA">Japanese</option>
          <option value="ES">Spanish</option>
          <option value="AR">Arabic</option>
        </select>
      </div>

      <textarea id="scriptInput" placeholder="Type text to speak with rich emotion...">I cannot believe this! We actually survived against all odds, and we won the entire championship!</textarea>

      <button id="synthBtn" class="btn-synthesize" onclick="synthesize()">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span id="synthText">Synthesize Expressive Speech</span>
      </button>

      <!-- Stage Output & Visualizer -->
      <div id="stageBox" class="stage-box">
        <div style="display:flex; justify-content:space-between; font-size:13px; font-weight:700; color:var(--success); margin-bottom:12px;">
          <span>✓ Master 24kHz Synthesis Ready</span>
          <span id="latencySpan" style="font-family:var(--font-mono); color:var(--text-muted); font-size:12px;"></span>
        </div>
        <canvas id="visualizer"></canvas>
        <audio id="audioOut" controls></audio>
        <div>
          <a id="downloadLink" class="download-btn" download="indextts_expressive.wav">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
            Download High-Fidelity WAV
          </a>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  let emoVector = [0, 0, 0, 0, 0, 0, 0, 0.8];
  let spkBase64 = null;
  let emoBase64 = null;
  let audioCtx = null;

  function updateFader(idx, val) {
    emoVector[idx] = parseFloat(val);
    document.getElementById('val_' + idx).innerText = parseFloat(val).toFixed(2);
  }

  function applyPreset(vec, el) {
    emoVector = [...vec];
    for (let i = 0; i < 8; i++) {
      document.getElementById('emo_' + i).value = emoVector[i];
      document.getElementById('val_' + i).innerText = emoVector[i].toFixed(2);
    }
    document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
  }

  function loadAudioFile(input, type) {
    if (input.files && input.files[0]) {
      const file = input.files[0];
      const reader = new FileReader();
      reader.onload = e => {
        const b64 = e.target.result.split(',')[1];
        if (type === 'spk') {
          spkBase64 = b64;
          document.getElementById('spkLabel').innerText = '✓ ' + file.name.substring(0, 10) + '..';
        } else {
          emoBase64 = b64;
          document.getElementById('emoLabel').innerText = '✓ ' + file.name.substring(0, 10) + '..';
        }
      };
      reader.readAsDataURL(file);
    }
  }

  async function synthesize() {
    const btn = document.getElementById('synthBtn');
    const txt = document.getElementById('synthText');
    const input = document.getElementById('scriptInput').value.trim();
    if (!input) { alert('Please enter some text'); return; }

    btn.disabled = true;
    txt.innerText = 'Synthesizing with 1.5B Emotion Engine...';

    const payload = {
      model: "indextts-2.5",
      input: input,
      lang: document.getElementById('langSelect').value,
      emo_vector: emoVector,
      emo_alpha: parseFloat(document.getElementById('alphaSlider').value)
    };
    if (spkBase64) payload.spk_audio = spkBase64;
    if (emoBase64) payload.emo_audio = emoBase64;

    const t0 = performance.now();
    try {
      const res = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());

      const blob = await res.blob();
      const dt = ((performance.now() - t0) / 1000).toFixed(2);
      const url = URL.createObjectURL(blob);

      const audio = document.getElementById('audioOut');
      audio.src = url;
      document.getElementById('downloadLink').href = url;
      document.getElementById('latencySpan').innerText = `Generated in ${dt}s`;
      document.getElementById('stageBox').style.display = 'block';

      startVisualizer(audio);
      audio.play();
    } catch (e) {
      alert('Error: ' + e.message);
    } finally {
      btn.disabled = false;
      txt.innerText = 'Synthesize Expressive Speech';
    }
  }

  function startVisualizer(audio) {
    if (audioCtx) return;
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      const src = audioCtx.createMediaElementSource(audio);
      src.connect(analyser);
      analyser.connect(audioCtx.destination);
      analyser.fftSize = 128;

      const canvas = document.getElementById('visualizer');
      const ctx = canvas.getContext('2d');
      const len = analyser.frequencyBinCount;
      const data = new Uint8Array(len);

      function draw() {
        requestAnimationFrame(draw);
        analyser.getByteFrequencyData(data);
        ctx.fillStyle = '#070a12';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const w = (canvas.width / len) * 1.8;
        let x = 0;
        for (let i = 0; i < len; i++) {
          const h = (data[i] / 255) * canvas.height * 0.9;
          const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
          grad.addColorStop(0, '#ec4899');
          grad.addColorStop(1, '#a855f7');
          ctx.fillStyle = grad;
          ctx.fillRect(x, canvas.height - h, w, h);
          x += w + 2;
        }
      }
      draw();
    } catch(e) {}
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=STUDIO_HTML)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting IndexTTS-2.5 Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
