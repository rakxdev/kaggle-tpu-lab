#!/usr/bin/env python3
"""
IndicF5 Production Master Server:
- Zero-Shot Voice Cloning with Shubh Persona (Warm Hindi Storyteller)
- OpenAI-compatible /v1/audio/speech API
- IndicXlit Unified Devanagari Normalizer (Eliminates Latin character stutter)
- Interactive IndicF5 Web Studio UI with Real-time Waveform Visualizer
- Concurrency queue lock to prevent GPU collisions
"""

import os
# Headless non-GUI backend to prevent Kaggle notebook matplotlib-inline collisions
os.environ["MPLBACKEND"] = "Agg"

import sys

# Auto-re-execute inside isolated Python 3.11 venv
VENV_PY = "/kaggle/tmp/indicf5/venv/bin/python"
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

# Add kit directory to sys.path for local modules
KIT_DIR = Path(__file__).resolve().parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import uvicorn
from fastapi import FastAPI, HTTPException, Header, status, UploadFile, File, Form
from fastapi.responses import Response, HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
# Allow smooth eager fallback if torch.compile meets SM75 kernel quirks
torch._dynamo.config.suppress_errors = True

import numpy as np
import soundfile as sf
import huggingface_hub
from transformers import AutoModel
from lib_normalize import to_unified_devanagari
from auth_helper import get_hf_token

BASE_DIR = Path("/kaggle/tmp/indicf5")
CHECKPOINTS = BASE_DIR / "checkpoints"
SHUBH_WAV = BASE_DIR / "shubh_ref.wav"
SHUBH_TXT = BASE_DIR / "shubh_ref.txt"

# Guard hf_hub_download to resolve local files in checkpoints directory
_orig_hf_download = huggingface_hub.hf_hub_download
def _safe_hf_download(repo_id, filename, **kwargs):
    cand = CHECKPOINTS / filename
    if cand.exists():
        return str(cand)
    cand_sub = CHECKPOINTS / "checkpoints" / filename
    if cand_sub.exists():
        return str(cand_sub)
    cand_base = CHECKPOINTS / Path(filename).name
    if cand_base.exists():
        return str(cand_base)
    return _orig_hf_download(repo_id, filename, **kwargs)
huggingface_hub.hf_hub_download = _safe_hf_download

# Fallback asset paths inside the git checkout if not yet copied to /kaggle/tmp
if not SHUBH_WAV.exists():
    fallback_wav = KIT_DIR / "assets" / "shubh_ref.wav"
    if fallback_wav.exists():
        SHUBH_WAV = fallback_wav

if not SHUBH_TXT.exists():
    fallback_txt = KIT_DIR / "assets" / "shubh_ref.txt"
    if fallback_txt.exists():
        SHUBH_TXT = fallback_txt

app = FastAPI(
    title="IndicF5 Master Voice Studio",
    description="SOTA Indic & Hinglish Zero-Shot Voice Cloning powered by AI4Bharat IndicF5 & IndicXlit.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_model = None
gpu_lock = asyncio.Lock()

def get_model():
    global tts_model
    if tts_model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[IndicF5] Loading model into {device}...")
        t0 = time.time()
        
        # Ensure HF token is loaded
        token = get_hf_token()
        if token:
            import huggingface_hub
            huggingface_hub.login(token=token, add_to_git_credential=False)
            
        model_source = str(CHECKPOINTS) if (CHECKPOINTS / "model.safetensors").exists() else "ai4bharat/IndicF5"
        tts_model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
        ).to(device)
        print(f"[IndicF5] Model initialized in {time.time() - t0:.2f}s")
    return tts_model

class SpeechRequest(BaseModel):
    model: str = Field(default="indicf5", description="Model identifier")
    input: str = Field(..., description="Text to synthesize (Devanagari or Hinglish)")
    voice: Optional[str] = Field(default="shubh", description="Voice preset: 'shubh' or 'custom'")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="Speaking speed rate")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav)")
    normalize: Optional[bool] = Field(default=True, description="Auto-transliterate Roman words to Devanagari")
    custom_ref_audio: Optional[str] = Field(default=None, description="Base64 encoded WAV audio for custom clone")
    custom_ref_text: Optional[str] = Field(default=None, description="Transcript of custom reference audio")

class NormalizeRequest(BaseModel):
    text: str = Field(..., description="Raw Hinglish text")

@app.get("/health")
async def health():
    vram_alloc = 0.0
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
    return {
        "status": "ok",
        "engine": "IndicF5 (330M Flow-Matching Diffusion)",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "vram_allocated_gb": round(vram_alloc, 2),
        "default_voice": "Shubh (Bulbul v3 Persona)",
        "sample_rate": 24000,
        "features": [
            "zero-shot-voice-cloning",
            "duration-patched-mode-a",
            "indicxlit-unified-devanagari",
            "openai-speech-v1"
        ]
    }

@app.get("/v1/audio/voices")
async def list_voices():
    return {
        "voices": [
            {
                "id": "shubh",
                "name": "Shubh (Hindi Storyteller & Reels Voiceover)",
                "gender": "male",
                "language": "hi-IN",
                "sample_rate": 24000,
                "description": "Warm, engaging, natural narrative voice from Bulbul v3 benchmark with studio cadence."
            },
            {
                "id": "custom",
                "name": "Custom Reference Voice",
                "gender": "variable",
                "language": "multilingual-indic",
                "sample_rate": 24000,
                "description": "Upload any 5-10 second clean audio clip to clone instantly."
            }
        ]
    }

@app.get("/api/reference_audio")
async def get_reference_audio():
    if SHUBH_WAV.exists():
        return FileResponse(str(SHUBH_WAV), media_type="audio/wav", filename="shubh_reference.wav")
    raise HTTPException(status_code=404, detail="Reference audio not found")

@app.post("/api/preview_normalize")
async def preview_normalize(req: NormalizeRequest):
    normalized = to_unified_devanagari(req.text)
    return {
        "raw": req.text,
        "normalized": normalized
    }

@app.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    # 1. Text Normalization
    target_text = req.input.strip()
    if req.normalize:
        target_text = to_unified_devanagari(target_text)
        if not target_text.strip():
            target_text = req.input.strip()

    # 2. Reference Audio Selection
    temp_wav_file = None
    try:
        if req.voice == "custom" and req.custom_ref_audio:
            # Decode base64 custom audio
            audio_bytes = base64.b64decode(req.custom_ref_audio)
            temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav_file.write(audio_bytes)
            temp_wav_file.flush()
            ref_audio_path = temp_wav_file.name
            ref_text = req.custom_ref_text or "custom prompt"
        else:
            if not SHUBH_WAV.exists() or not SHUBH_TXT.exists():
                raise HTTPException(status_code=500, detail="Default Shubh reference files not found on server.")
            ref_audio_path = str(SHUBH_WAV)
            ref_text = SHUBH_TXT.read_text(encoding="utf-8").strip()

        # 3. Model Inference with Concurrency Guard
        async with gpu_lock:
            model = get_model()
            t0 = time.time()
            with torch.inference_mode():
                raw_audio = model(
                    text=target_text,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                )
            latency = time.time() - t0

        # 4. Format Output Array
        if hasattr(raw_audio, "cpu"):
            raw_audio = raw_audio.cpu().numpy()
        audio_arr = np.asarray(raw_audio, dtype=np.float32).squeeze()
        
        if raw_audio.dtype == np.int16 or np.max(np.abs(audio_arr)) > 1.5:
            audio_arr = audio_arr.astype(np.float32) / 32768.0

        buf = io.BytesIO()
        sf.write(buf, audio_arr, 24000, format="WAV")
        buf.seek(0)
        audio_bytes = buf.read()
        duration = len(audio_arr) / 24000.0

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=synthesized_shubh.wav",
                "X-Latency-Seconds": f"{latency:.3f}",
                "X-Audio-Duration": f"{duration:.3f}",
                "X-Normalized-Text": target_text.encode("utf-8").decode("latin-1", errors="replace"),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if temp_wav_file and os.path.exists(temp_wav_file.name):
            try:
                os.remove(temp_wav_file.name)
            except Exception:
                pass

STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IndicF5 Master Voice Studio — Kaggle T4</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; color: #f1f5f9; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .glass { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .accent-gradient { background: linear-gradient(135deg, #f59e0b 0%, #ea580c 50%, #e11d48 100%); }
    .border-glow:focus-within { border-color: #f59e0b; box-shadow: 0 0 20px rgba(245, 158, 11, 0.25); }
  </style>
</head>
<body class="min-h-screen flex flex-col justify-between selection:bg-amber-500/30">

  <!-- Top Header Navigation -->
  <header class="border-b border-white/10 glass sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl accent-gradient flex items-center justify-center shadow-lg shadow-amber-500/20 font-extrabold text-white text-lg">
          आई
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="font-bold text-lg text-white">IndicF5 Master Studio</h1>
            <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">Shubh Persona</span>
          </div>
          <p class="text-xs text-slate-400">Zero-Shot Hinglish Voice Cloning • AI4Bharat IndicF5 (330M) • Kaggle T4</p>
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <div id="gpuStatusBadge" class="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span id="gpuStatusText">GPU 0 • 24 kHz Live</span>
        </div>
        <a href="/docs" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition border border-white/10">API Docs</a>
      </div>
    </div>
  </header>

  <!-- Main Studio Layout -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
    
    <!-- Left Column: Controls & Text Input (7 Cols) -->
    <div class="lg:col-span-7 space-y-6">
      
      <!-- Voice Persona Selection Card -->
      <div class="glass rounded-2xl p-6 space-y-4">
        <div class="flex items-center justify-between">
          <label class="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center space-x-2">
            <span>1. Voice Persona</span>
          </label>
          <button id="btnListenRef" class="text-xs text-amber-400 hover:text-amber-300 flex items-center space-x-1.5 py-1 px-2.5 rounded-md bg-amber-500/10 hover:bg-amber-500/20 transition">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <span>Listen Reference Sample</span>
          </button>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <label class="relative flex flex-col p-4 rounded-xl border border-amber-500/50 bg-amber-500/5 cursor-pointer hover:border-amber-400 transition" id="cardShubh">
            <input type="radio" name="voiceSelect" value="shubh" checked class="sr-only">
            <div class="flex items-center justify-between mb-1">
              <span class="font-bold text-white text-sm">Shubh (Bulbul v3)</span>
              <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
            </div>
            <p class="text-xs text-slate-400">Warm narrative Hindi voice. Ideal for YouTube Shorts, Reels, and storytelling.</p>
          </label>

          <label class="relative flex flex-col p-4 rounded-xl border border-white/10 bg-white/5 cursor-pointer hover:border-white/20 transition opacity-80" id="cardCustom">
            <input type="radio" name="voiceSelect" value="custom" class="sr-only">
            <div class="flex items-center justify-between mb-1">
              <span class="font-bold text-white text-sm">Custom Audio Upload</span>
              <span class="w-2.5 h-2.5 rounded-full border border-slate-500"></span>
            </div>
            <p class="text-xs text-slate-400">Clone your own voice from a 5–10s WAV audio sample.</p>
          </label>
        </div>

        <!-- Custom Upload Section (Hidden unless Custom is selected) -->
        <div id="customVoiceSection" class="hidden space-y-3 pt-3 border-t border-white/10">
          <div>
            <label class="block text-xs text-slate-400 mb-1">Upload Reference WAV (5–10 seconds clean mono)</label>
            <input type="file" id="customAudioFile" accept="audio/*" class="block w-full text-xs text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-amber-500 file:text-slate-900 hover:file:bg-amber-400 cursor-pointer">
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Transcript of the uploaded audio clip</label>
            <input type="text" id="customAudioText" placeholder="Exact words spoken in the reference audio" class="w-full bg-slate-900 border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-amber-500">
          </div>
        </div>
      </div>

      <!-- Script Input Section -->
      <div class="glass rounded-2xl p-6 space-y-4">
        <div class="flex items-center justify-between">
          <label class="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center space-x-2">
            <span>2. Script to Synthesize</span>
          </label>
          <div class="flex items-center space-x-2">
            <span class="text-xs text-slate-400" id="charCount">0 characters</span>
          </div>
        </div>

        <!-- Quick Sample Presets -->
        <div class="flex flex-wrap gap-2 text-xs">
          <button class="px-2.5 py-1 rounded-md bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 transition preset-btn" data-text="बारिश की एक ठंडी शाम थी। शहर की एक छोटी सी चाय की दुकान पर मैं अपनी चाय का इंतज़ार कर रहा था।">
            📖 Classic Story
          </button>
          <button class="px-2.5 py-1 rounded-md bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 transition preset-btn" data-text="Yaar, sach bataun toh hum kitna zyada bhagte hain roz. Thoda relax karo, sab theek ho jayega.">
            ⚡ Hinglish Reel
          </button>
          <button class="px-2.5 py-1 rounded-md bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 transition preset-btn" data-text="Kal office mein new project launch ki meeting hai. Poori presentation ready hai, tension bilkul mat lo.">
            💼 Tech / Office
          </button>
        </div>

        <div class="border-glow rounded-xl border border-white/10 transition bg-slate-900/60 p-3">
          <textarea id="textInput" rows="5" class="w-full bg-transparent text-white placeholder-slate-500 focus:outline-none text-base resize-none" placeholder="Type your text in Devanagari or Romanized Hinglish (e.g., 'Kal office mein meeting hai')..."></textarea>
        </div>

        <!-- Live Transliteration Preview -->
        <div class="p-3.5 rounded-xl bg-slate-950/80 border border-white/10 space-y-1">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold text-amber-400 flex items-center space-x-1.5">
              <span>Devanagari Acoustic Engine Input:</span>
            </span>
            <span class="text-[10px] text-slate-400">IndicXlit Unified Output</span>
          </div>
          <p id="normalizedPreview" class="text-xs text-slate-300 font-mono italic">Start typing to see phonetic transliteration...</p>
        </div>

        <!-- Settings Bar -->
        <div class="flex flex-wrap items-center justify-between pt-2 gap-4">
          <div class="flex items-center space-x-4">
            <label class="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
              <input type="checkbox" id="chkNormalize" checked class="rounded bg-slate-900 border-white/20 text-amber-500 focus:ring-amber-400 w-4 h-4">
              <span>Auto-Normalize Hinglish</span>
            </label>
          </div>

          <div class="flex items-center space-x-3">
            <span class="text-xs text-slate-400">Speed:</span>
            <input type="range" id="speedSlider" min="0.8" max="1.3" step="0.02" value="1.0" class="w-24 accent-amber-500 cursor-pointer">
            <span id="speedVal" class="text-xs font-mono text-slate-200">1.00x</span>
          </div>
        </div>

        <!-- Action Button -->
        <button id="btnSynthesize" class="w-full py-4 rounded-xl accent-gradient text-slate-950 font-extrabold text-base shadow-lg shadow-amber-500/20 hover:opacity-95 active:scale-[0.99] transition flex items-center justify-center space-x-2">
          <svg id="synthSpinner" class="hidden w-5 h-5 animate-spin text-slate-950" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
          <span id="btnSynthText">Generate Cloned Speech (Ctrl + Enter)</span>
        </button>
      </div>
    </div>

    <!-- Right Column: Visualizer, Audio Player & Code Snippets (5 Cols) -->
    <div class="lg:col-span-5 space-y-6">
      
      <!-- Visualizer & Player Card -->
      <div class="glass rounded-2xl p-6 space-y-5">
        <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center justify-between">
          <span>Master Output & Waveform</span>
          <span id="metricDuration" class="text-xs text-amber-400 font-mono">0.00s</span>
        </h2>

        <!-- Canvas Spectrum Visualizer -->
        <div class="relative h-36 w-full rounded-xl bg-slate-950 border border-white/10 overflow-hidden flex items-center justify-center">
          <canvas id="visualizerCanvas" class="w-full h-full"></canvas>
          <div id="visIdleText" class="absolute text-xs text-slate-600 font-mono flex items-center space-x-2">
            <span>● 24,000 Hz BigVGAN Audio Output Canvas</span>
          </div>
        </div>

        <!-- Main Audio Player Element -->
        <div class="space-y-3">
          <audio id="audioPlayer" controls class="w-full rounded-lg"></audio>
          
          <div class="flex items-center justify-between text-xs text-slate-400 pt-1 font-mono">
            <span id="metricLatency">Latency: --</span>
            <span id="metricRtf">RTF: --</span>
            <button id="btnDownload" class="hidden text-amber-400 hover:underline flex items-center space-x-1">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
              <span>Download WAV</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Quick Developer Integration -->
      <div class="glass rounded-2xl p-6 space-y-4">
        <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">Developer Integration</h2>
        <p class="text-xs text-slate-400">OpenAI-compatible <code class="text-amber-300">/v1/audio/speech</code> endpoint:</p>
        
        <div class="bg-slate-950 p-3.5 rounded-xl border border-white/10 overflow-x-auto text-[11px] font-mono text-slate-300 leading-relaxed">
          <span class="text-amber-400">curl</span> -X POST /v1/audio/speech \<br>
          &nbsp;&nbsp;-H <span class="text-emerald-300">"Content-Type: application/json"</span> \<br>
          &nbsp;&nbsp;-d '<span class="text-amber-200">{"input": "कल office में meeting है।", "voice": "shubh"}</span>' \<br>
          &nbsp;&nbsp;--output output.wav
        </div>
      </div>
    </div>
  </main>

  <!-- Audio element for playing reference clip -->
  <audio id="refAudioPlayer" class="hidden" src="/api/reference_audio"></audio>

  <footer class="border-t border-white/5 py-4 text-center text-xs text-slate-500">
    AI4Bharat IndicF5 (330M) • IndicXlit Normalizer • Zero-Shot Voice Cloning on Kaggle T4 Dual GPU
  </footer>

  <script>
    const textInput = document.getElementById('textInput');
    const normalizedPreview = document.getElementById('normalizedPreview');
    const charCount = document.getElementById('charCount');
    const speedSlider = document.getElementById('speedSlider');
    const speedVal = document.getElementById('speedVal');
    const btnSynthesize = document.getElementById('btnSynthesize');
    const btnSynthText = document.getElementById('btnSynthText');
    const synthSpinner = document.getElementById('synthSpinner');
    const audioPlayer = document.getElementById('audioPlayer');
    const refAudioPlayer = document.getElementById('refAudioPlayer');
    const btnListenRef = document.getElementById('btnListenRef');
    const btnDownload = document.getElementById('btnDownload');
    const metricDuration = document.getElementById('metricDuration');
    const metricLatency = document.getElementById('metricLatency');
    const metricRtf = document.getElementById('metricRtf');
    const chkNormalize = document.getElementById('chkNormalize');
    const canvas = document.getElementById('visualizerCanvas');
    const canvasCtx = canvas.getContext('2d');
    const visIdleText = document.getElementById('visIdleText');
    
    // Voice selection elements
    const cardShubh = document.getElementById('cardShubh');
    const cardCustom = document.getElementById('cardCustom');
    const customVoiceSection = document.getElementById('customVoiceSection');
    const customAudioFile = document.getElementById('customAudioFile');
    const customAudioText = document.getElementById('customAudioText');

    let currentVoice = 'shubh';
    let currentAudioUrl = null;
    let customBase64Audio = null;

    // Handle voice switching
    document.querySelectorAll('input[name="voiceSelect"]').forEach(radio => {
      radio.addEventListener('change', (e) => {
        currentVoice = e.target.value;
        if (currentVoice === 'custom') {
          customVoiceSection.classList.remove('hidden');
          cardCustom.classList.add('border-amber-500/50', 'bg-amber-500/5');
          cardCustom.classList.remove('border-white/10', 'bg-white/5', 'opacity-80');
          cardShubh.classList.remove('border-amber-500/50', 'bg-amber-500/5');
          cardShubh.classList.add('border-white/10', 'bg-white/5', 'opacity-80');
        } else {
          customVoiceSection.classList.add('hidden');
          cardShubh.classList.add('border-amber-500/50', 'bg-amber-500/5');
          cardShubh.classList.remove('border-white/10', 'bg-white/5', 'opacity-80');
          cardCustom.classList.remove('border-amber-500/50', 'bg-amber-500/5');
          cardCustom.classList.add('border-white/10', 'bg-white/5', 'opacity-80');
        }
      });
    });

    // Custom audio file input to base64
    customAudioFile.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
          customBase64Audio = evt.target.result.split(',')[1];
        };
        reader.readAsDataURL(file);
      }
    });

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        textInput.value = btn.dataset.text;
        updatePreview();
      });
    });

    // Speed slider
    speedSlider.addEventListener('input', (e) => {
      speedVal.innerText = parseFloat(e.target.value).toFixed(2) + 'x';
    });

    // Reference audio playback toggle
    btnListenRef.addEventListener('click', () => {
      if (refAudioPlayer.paused) {
        refAudioPlayer.play();
        btnListenRef.innerHTML = '<span>⏸ Pause Sample</span>';
      } else {
        refAudioPlayer.pause();
        btnListenRef.innerHTML = '<span>▶ Listen Reference Sample</span>';
      }
    });
    refAudioPlayer.onended = () => {
      btnListenRef.innerHTML = '<span>▶ Listen Reference Sample</span>';
    };

    // Live preview debounce
    let previewTimer = null;
    function updatePreview() {
      const val = textInput.value;
      charCount.innerText = val.length + ' characters';
      clearTimeout(previewTimer);
      if (!val.trim()) {
        normalizedPreview.innerText = "Start typing to see phonetic transliteration...";
        return;
      }
      previewTimer = setTimeout(async () => {
        try {
          const res = await fetch('/api/preview_normalize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: val})
          });
          if (res.ok) {
            const data = await res.json();
            normalizedPreview.innerText = data.normalized;
          }
        } catch (err) {}
      }, 350);
    }
    textInput.addEventListener('input', updatePreview);

    // Audio Canvas Visualizer Setup
    let audioCtx, analyser, source, animationId;
    function setupVisualizer() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 128;
        source = audioCtx.createMediaElementSource(audioPlayer);
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
      }
    }

    function drawVisualizer() {
      animationId = requestAnimationFrame(drawVisualizer);
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);

      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.85;
        const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, '#f59e0b');
        gradient.addColorStop(0.5, '#ea580c');
        gradient.addColorStop(1, '#e11d48');

        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }
    }

    audioPlayer.onplay = () => {
      setupVisualizer();
      visIdleText.classList.add('hidden');
      if (audioCtx.state === 'suspended') audioCtx.resume();
      drawVisualizer();
    };
    audioPlayer.onpause = () => {
      cancelAnimationFrame(animationId);
    };
    audioPlayer.onended = () => {
      cancelAnimationFrame(animationId);
      visIdleText.classList.remove('hidden');
    };

    // Synthesis Handler
    async function doSynthesize() {
      const text = textInput.value.trim();
      if (!text) {
        alert("Please enter text to synthesize.");
        return;
      }

      btnSynthesize.disabled = true;
      synthSpinner.classList.remove('hidden');
      btnSynthText.innerText = "Synthesizing Diffusion Frames...";

      const payload = {
        model: "indicf5",
        input: text,
        voice: currentVoice,
        speed: parseFloat(speedSlider.value),
        normalize: chkNormalize.checked,
        custom_ref_audio: customBase64Audio,
        custom_ref_text: customAudioText.value.trim()
      };

      const t0 = performance.now();
      try {
        const res = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Synthesis failed");
        }

        const blob = await res.blob();
        const latency = (performance.now() - t0) / 1000;
        const durHeader = res.headers.get('X-Audio-Duration');
        const duration = durHeader ? parseFloat(durHeader) : 0;
        const rtf = duration > 0 ? (latency / duration).toFixed(3) : '--';

        if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = URL.createObjectURL(blob);
        audioPlayer.src = currentAudioUrl;
        audioPlayer.play();

        metricDuration.innerText = duration.toFixed(2) + 's';
        metricLatency.innerText = 'Latency: ' + latency.toFixed(2) + 's';
        metricRtf.innerText = 'RTF: ' + rtf + 'x';

        btnDownload.classList.remove('hidden');
        btnDownload.onclick = () => {
          const a = document.createElement('a');
          a.href = currentAudioUrl;
          a.download = 'shubh_cloned_' + Date.now() + '.wav';
          a.click();
        };

      } catch (err) {
        alert("Error: " + err.message);
      } finally {
        btnSynthesize.disabled = false;
        synthSpinner.classList.add('hidden');
        btnSynthText.innerText = "Generate Cloned Speech (Ctrl + Enter)";
      }
    }

    btnSynthesize.addEventListener('click', doSynthesize);
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        doSynthesize();
      }
    });

    // Default initialization
    window.addEventListener('load', () => {
      textInput.value = "बारिश की एक ठंडी शाम थी। शहर की एक छोटी सी चाय की दुकान पर मैं अपनी चाय का इंतज़ार कर रहा था।";
      updatePreview();
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_studio():
    return HTMLResponse(content=STUDIO_HTML)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="IndicF5 Production Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    # Pre-warm model in background
    get_model()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
