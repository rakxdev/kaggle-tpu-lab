#!/usr/bin/env python3
"""
F5-TTS Production Server:
- OpenAI-compatible /v1/audio/speech API
- Preset voices & zero-shot cloning endpoint
- Responsive Web Studio UI for mobile & desktop
- GPU queue lock to prevent concurrent OOM collisions
"""

import os
import sys
import io
import time
import json
import base64
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request, status
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Constants & Paths
BASE_DIR = Path("/kaggle/tmp/f5tts")
MODEL_DIR = BASE_DIR / "models"
VOICES_DIR = BASE_DIR / "voices"
MANIFEST_PATH = VOICES_DIR / "voices.json"
API_KEY = os.environ.get("F5_API_KEY", "")

app = FastAPI(
    title="F5-TTS Voice Studio & OpenAI Speech API",
    description="Zero-shot voice cloning and high-fidelity text-to-speech engine powered by F5-TTS on NVIDIA T4.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model holder & GPU concurrency lock
f5_model = None
gpu_lock = asyncio.Lock()
manifest: Dict[str, Any] = {}

def get_model():
    global f5_model
    if f5_model is None:
        import torch
        from f5_tts.api import F5TTS

        ckpt_file = MODEL_DIR / "F5TTS_v1_Base" / "model_1250000.safetensors"
        vocab_file = MODEL_DIR / "F5TTS_v1_Base" / "vocab.txt"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"[F5-TTS] Loading model into {device}...")
        t0 = time.time()
        f5_model = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=str(ckpt_file) if ckpt_file.exists() else "",
            vocab_file=str(vocab_file) if vocab_file.exists() else "",
            device=device,
        )
        print(f"[F5-TTS] Model loaded in {time.time() - t0:.2f}s")
    return f5_model

def load_manifest():
    global manifest
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"[F5-TTS] Failed to read manifest: {e}")
    if not manifest:
        manifest = {
            "nature_en": {
                "file": "/kaggle/tmp/f5tts/voices/nature_en.wav",
                "transcript": "Some call me nature, others call me mother nature.",
                "description": "Warm Nature / Documentary (English)",
            }
        }
    return manifest

# Authorization check
def verify_auth(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected Bearer token.",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

# Request Models
class SpeechRequest(BaseModel):
    model: str = Field(default="f5-tts", description="Model name (e.g. f5-tts, tts-1)")
    input: str = Field(..., description="The text to generate audio for")
    voice: Optional[str] = Field(default="nature_en", description="Preset voice ID, or voice mapping")
    response_format: Optional[str] = Field(default="wav", description="Audio format: wav")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Speed of generated audio")
    # Extended zero-shot cloning parameters
    ref_audio: Optional[str] = Field(default=None, description="Optional base64 WAV or path for zero-shot cloning")
    ref_text: Optional[str] = Field(default=None, description="Optional transcript of the reference audio")
    nfe_step: Optional[int] = Field(default=32, ge=8, le=64, description="Flow matching ODE steps (default 32)")

# OpenAI Voice Name Aliases
OPENAI_VOICE_MAP = {
    "alloy": "nature_en",
    "echo": "storyteller",
    "fable": "narrator",
    "onyx": "storyteller",
    "nova": "nature_en",
    "shimmer": "mandarin_zh",
}

@app.on_event("startup")
async def startup_event():
    load_manifest()
    # Eagerly load model in background worker
    asyncio.get_event_loop().run_in_executor(None, get_model)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "F5TTS_v1_Base",
        "voices_available": list(manifest.keys()),
        "auth_required": bool(API_KEY),
    }

@app.get("/v1/audio/voices")
def list_voices():
    load_manifest()
    return {"voices": manifest}

@app.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest, authorization: Optional[str] = Header(None)):
    verify_auth(authorization)

    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Text 'input' cannot be empty.")

    load_manifest()
    voice_key = req.voice or "nature_en"
    if voice_key in OPENAI_VOICE_MAP:
        voice_key = OPENAI_VOICE_MAP[voice_key]

    ref_file = None
    ref_text = ""
    temp_ref_path = None

    # Handle custom zero-shot audio upload in request
    if req.ref_audio:
        try:
            # Check if base64 encoded
            audio_bytes = base64.b64decode(req.ref_audio)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            temp_ref_path = tmp.name
            ref_file = temp_ref_path
            ref_text = req.ref_text or ""
        except Exception:
            # Maybe path on disk
            if os.path.exists(req.ref_audio):
                ref_file = req.ref_audio
                ref_text = req.ref_text or ""

    # If no custom ref audio supplied, use preset
    if not ref_file:
        preset = manifest.get(voice_key)
        if not preset:
            # Default fallback
            preset = manifest.get("nature_en") or next(iter(manifest.values()), None)

        if preset and os.path.exists(preset["file"]):
            ref_file = preset["file"]
            ref_text = req.ref_text or preset.get("transcript", "")
        else:
            from importlib.resources import files
            ref_file = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
            ref_text = req.ref_text or "Some call me nature, others call me mother nature."

    # Acquire GPU lock for sequential diffusion inference
    async with gpu_lock:
        try:
            model = get_model()
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_out_path = tmp_out.name
            tmp_out.close()

            def run_inference():
                return model.infer(
                    ref_file=ref_file,
                    ref_text=ref_text,
                    gen_text=req.input.strip(),
                    file_wave=tmp_out_path,
                    nfe_step=req.nfe_step or 32,
                    speed=req.speed or 1.0,
                )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_inference)

            with open(tmp_out_path, "rb") as f:
                wav_bytes = f.read()

            try:
                os.remove(tmp_out_path)
            except OSError:
                pass

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        finally:
            if temp_ref_path and os.path.exists(temp_ref_path):
                try:
                    os.remove(temp_ref_path)
                except OSError:
                    pass

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="speech.wav"',
            "X-Model": "F5TTS_v1_Base",
            "X-Voice": voice_key,
        },
    )

# Interactive Studio UI
DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>F5-TTS Voice Studio — Kaggle T4</title>
<style>
  :root {
    --bg: #090d16;
    --card: #131b2e;
    --border: #24324f;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --success: #10b981;
    --code-bg: #0b1120;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
  .container { width: 100%; max-width: 820px; }
  header { text-align: center; margin-bottom: 32px; }
  h1 { font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 8px; }
  .badge { display: inline-block; padding: 4px 12px; background: rgba(56, 189, 248, 0.15); color: var(--accent); border-radius: 999px; font-size: 13px; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.3); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
  label { display: block; font-size: 14px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px; }
  textarea, input[type="text"], select { width: 100%; background: var(--code-bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; color: #fff; font-size: 15px; outline: none; transition: border-color 0.2s; }
  textarea:focus, input[type="text"]:focus, select:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 110px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }
  @media(max-width: 600px) { .row { grid-template-columns: 1fr; } }
  .slider-row { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
  .slider-row input[type="range"] { flex: 1; accent-color: var(--accent); }
  button { width: 100%; background: var(--accent); color: #04101d; border: none; border-radius: 12px; padding: 15px; font-size: 16px; font-weight: 700; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 14px rgba(56, 189, 248, 0.35); }
  button:hover { background: var(--accent-hover); color: #fff; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .audio-player-box { margin-top: 24px; padding: 18px; background: var(--code-bg); border-radius: 12px; border: 1px solid var(--border); display: none; }
  audio { width: 100%; height: 48px; outline: none; margin-top: 8px; border-radius: 8px; }
  .api-info { margin-top: 16px; font-size: 13px; color: var(--text-muted); line-height: 1.6; }
  pre { background: var(--code-bg); padding: 14px; border-radius: 10px; overflow-x: auto; font-family: monospace; font-size: 13px; border: 1px solid var(--border); color: #38bdf8; margin-top: 8px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>F5-TTS Voice Studio</h1>
    <div class="badge">NVIDIA T4 · Flow Matching Diffusion · Zero-Shot Cloning</div>
  </header>

  <div class="card">
    <div class="row">
      <div>
        <label for="voiceSelect">Preset Voice</label>
        <select id="voiceSelect" onchange="onVoiceChange()">
          <option value="nature_en">English (Warm Documentary)</option>
          <option value="mandarin_zh">Mandarin (Clear Conversational)</option>
          <option value="storyteller">English (Storyteller / Audiobooks)</option>
          <option value="narrator">English (Calm Narrator)</option>
          <option value="custom">Upload Custom Voice (5-10s Audio)</option>
        </select>
      </div>
      <div>
        <label for="apiKey">API Key (Optional)</label>
        <input type="text" id="apiKey" placeholder="Leave empty if none configured">
      </div>
    </div>

    <div id="customVoiceBox" style="display:none; margin-bottom: 18px; padding: 16px; background: rgba(56, 189, 248, 0.05); border: 1px dashed var(--accent); border-radius: 10px;">
      <label for="refAudioFile">Upload 5–10s Reference Audio (WAV / MP3)</label>
      <input type="file" id="refAudioFile" accept="audio/*" style="margin-bottom: 10px;">
      <label for="refTranscript">Reference Audio Transcript (Exact words in clip)</label>
      <input type="text" id="refTranscript" placeholder="e.g. The exact words spoken in the audio clip">
    </div>

    <div style="margin-bottom: 18px;">
      <label for="inputText">Text to Synthesize</label>
      <textarea id="inputText" placeholder="Enter text to speak...">Welcome to the next generation of voice synthesis. F5-TTS reproduces realistic human timbre, prosody, and breath with flow matching diffusion.</textarea>
    </div>

    <div class="slider-row">
      <label style="margin: 0; min-width: 100px;">Speed: <span id="speedVal">1.0x</span></label>
      <input type="range" id="speedSlider" min="0.5" max="1.8" step="0.1" value="1.0" oninput="document.getElementById('speedVal').innerText = this.value + 'x'">
    </div>

    <button id="genBtn" onclick="generateAudio()">Generate Speech</button>

    <div id="playerBox" class="audio-player-box">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
        <span style="font-size:14px; font-weight:600; color:var(--success);">✓ Audio Generated</span>
        <span id="genTime" style="font-size:13px; color:var(--text-muted);"></span>
      </div>
      <audio id="audioElement" controls></audio>
    </div>
  </div>

  <div class="card">
    <label>OpenAI-Compatible Speech API</label>
    <div class="api-info">
      Use this endpoint in any OpenAI-compatible app, mobile browser, or Python client:
    </div>
    <pre id="curlSnippet">curl -X POST &lt;ORIGIN&gt;/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{"model":"f5-tts","input":"Hello from F5-TTS!","voice":"nature_en"}' \\
  --output speech.wav</pre>
  </div>
</div>

<script>
  // Dynamically update snippet with current origin
  document.getElementById('curlSnippet').innerText = 
    `curl -X POST ${window.location.origin}/v1/audio/speech \\\\\n` +
    `  -H "Content-Type: application/json" \\\\\n` +
    `  -d '{"model":"f5-tts","input":"Hello from F5-TTS!","voice":"nature_en"}' \\\\\n` +
    `  --output speech.wav`;

  function onVoiceChange() {
    const val = document.getElementById('voiceSelect').value;
    document.getElementById('customVoiceBox').style.display = (val === 'custom') ? 'block' : 'none';
  }

  async function generateAudio() {
    const btn = document.getElementById('genBtn');
    const playerBox = document.getElementById('playerBox');
    const audio = document.getElementById('audioElement');
    const genTime = document.getElementById('genTime');
    const text = document.getElementById('inputText').value.trim();
    const voice = document.getElementById('voiceSelect').value;
    const speed = parseFloat(document.getElementById('speedSlider').value);
    const key = document.getElementById('apiKey').value.trim();

    if (!text) { alert('Please enter some text'); return; }

    btn.disabled = true;
    btn.innerText = 'Synthesizing with Flow Matching Diffusion...';

    const payload = {
      model: "f5-tts",
      input: text,
      voice: voice,
      speed: speed
    };

    if (voice === 'custom') {
      const fileInput = document.getElementById('refAudioFile');
      if (!fileInput.files.length) {
        alert('Please choose a reference audio file');
        btn.disabled = false;
        btn.innerText = 'Generate Speech';
        return;
      }
      const refText = document.getElementById('refTranscript').value.trim();
      const file = fileInput.files[0];
      const reader = new FileReader();
      const b64 = await new Promise((resolve) => {
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(file);
      });
      payload.ref_audio = b64;
      payload.ref_text = refText;
    }

    const t0 = performance.now();
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (key) headers['Authorization'] = 'Bearer ' + key;

      const resp = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Inference failed');
      }

      const blob = await resp.blob();
      const dt = ((performance.now() - t0) / 1000).toFixed(2);
      audio.src = URL.createObjectURL(blob);
      playerBox.style.display = 'block';
      genTime.innerText = `Generated in ${dt}s`;
      audio.play();
    } catch (e) {
      alert('Error: ' + e.message);
    } finally {
      btn.disabled = false;
      btn.innerText = 'Generate Speech';
    }
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=DOCS_HTML)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--api-key", type=str, default="")
    args = parser.parse_args()

    if args.api_key:
        API_KEY = args.api_key
        os.environ["F5_API_KEY"] = args.api_key

    print(f"Starting F5-TTS FastAPI Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
