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
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add kit directory to sys.path for local modules
KIT_DIR = Path(__file__).resolve().parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

# Auto-align dependency matrix: datasets 2.21.0 + pyarrow 18.0.0 for NumPy 1.x & HF Hub compatibility
import subprocess
_needs_align = []
try:
    import pyarrow
    if hasattr(pyarrow, "__version__") and int(pyarrow.__version__.split(".")[0]) >= 19:
        _needs_align.append("pyarrow==18.0.0")
except Exception:
    _needs_align.append("pyarrow==18.0.0")

try:
    import datasets
    if hasattr(datasets, "__version__") and int(datasets.__version__.split(".")[0]) >= 3:
        _needs_align.append("datasets==2.21.0")
except Exception:
    _needs_align.append("datasets==2.21.0")

if _needs_align:
    print(f"Aligning dependency matrix ({', '.join(_needs_align)})...")
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "-q"] + _needs_align, check=False)
    os.execv(sys.executable, [sys.executable] + sys.argv)

def patch_f5_tts_engine():
    """Decouples training dependencies and ensures duration patch on all copies."""
    import glob
    # 1. Patch utils_infer.py: direct CFM import & duration fix
    for p in glob.glob("/kaggle/tmp/indicf5/**/utils_infer.py", recursive=True) + \
             glob.glob("/root/.cache/huggingface/**/utils_infer.py", recursive=True):
        try:
            s = Path(p).read_text(encoding="utf-8")
            c = False
            if "from f5_tts.model import CFM" in s:
                s = s.replace("from f5_tts.model import CFM", "from f5_tts.model.cfm import CFM")
                c = True
            if 'len(ref_text.encode("utf-8"))' in s:
                s = s.replace('len(ref_text.encode("utf-8"))', 'sum(1 for c in ref_text if not c.isspace())')
                s = s.replace('len(gen_text.encode("utf-8"))', 'sum(1 for c in gen_text if not c.isspace())')
                c = True
            bad_block = "try:\n            vocoder = vocoder.eval().to(device)\n        except Exception:\n            vocoder = vocoder.to_empty(device=device).eval()"
            if bad_block in s:
                s = s.replace(bad_block, "vocoder = vocoder.eval().to(device)")
                c = True

            # 3. Patch chunk_text: split on Devanagari । as well as Latin punctuation
            if 'r"(?<=[;:,.!?])\\s+|(?<=[；：，。！？])"' in s:
                s = s.replace('r"(?<=[;:,.!?])\\s+|(?<=[；：，。！？])"', 'r"(?<=[;:,.!?।])\\s+|(?<=[；：，।。！？])"')
                c = True
            if 'if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:' in s:
                s = s.replace(
                    'if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:',
                    'if len(current_chunk) + len(sentence) <= max_chars:'
                )
                c = True
            # 4. Cap max_chars to 100 characters per chunk to prevent DiT attention context overflow
            for line in s.splitlines():
                if "max_chars = int(" in line:
                    s = s.replace(line, "    max_chars = 100")
                    c = True

            # 5. Fast high-quality diffusion: 18 ODE steps instead of 32 (2x speedup)
            if "nfe_step=32" in s:
                s = s.replace("nfe_step=32", "nfe_step=18")
                c = True
            if c:
                Path(p).write_text(s, encoding="utf-8")
        except Exception:
            pass

    # 2. Patch model/__init__.py: wrap Trainer in try/except
    for p in glob.glob("/kaggle/tmp/indicf5/**/model/__init__.py", recursive=True) + \
             glob.glob("/root/.cache/huggingface/**/model/__init__.py", recursive=True):
        try:
            s = Path(p).read_text(encoding="utf-8")
            if "from f5_tts.model.trainer import Trainer" in s and "try:" not in s:
                s = s.replace(
                    "from f5_tts.model.trainer import Trainer",
                    "try:\n    from f5_tts.model.trainer import Trainer\nexcept Exception:\n    Trainer = None"
                )
                Path(p).write_text(s, encoding="utf-8")
        except Exception:
            pass

patch_f5_tts_engine()

import uvicorn
from fastapi import FastAPI, HTTPException, Header, status, UploadFile, File, Form
from fastapi.responses import Response, HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import torch
# Allow smooth eager fallback if torch.compile meets SM75 kernel quirks
torch._dynamo.config.suppress_errors = True

# Disable Accelerate's init_empty_weights context so Transformers instantiates submodules
# (like Vocos vocoder) with real device memory rather than empty meta tensors
import contextlib
try:
    import accelerate
    @contextlib.contextmanager
    def _noop_context(*args, **kwargs):
        yield
    accelerate.init_empty_weights = _noop_context
except Exception:
    pass

import transformers
import transformers.utils.import_utils
transformers.utils.import_utils.is_accelerate_available = lambda: False
try:
    import transformers.modeling_utils
    transformers.modeling_utils.init_empty_weights = _noop_context
except Exception:
    pass

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

WEB_DIR = KIT_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

class DualGPUWorkerPool:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.devices = []
        self._initialized = False

    def init_workers(self):
        if self._initialized:
            return
        t0_all = time.time()
        token = get_hf_token()
        if token:
            import huggingface_hub
            huggingface_hub.login(token=token, add_to_git_credential=False)

        model_source = str(CHECKPOINTS) if (CHECKPOINTS / "model.safetensors").exists() else "ai4bharat/IndicF5"

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            target_devices = [f"cuda:{i}" for i in range(min(count, 2))]
        else:
            target_devices = ["cpu"]

        print(f"[IndicF5 Dual-GPU Pool] Initializing {len(target_devices)} parallel workers on {target_devices}...")
        for dev in target_devices:
            print(f"  [Pool Worker] Loading model into {dev}...")
            t0 = time.time()
            model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            ).to(dev)
            self.queue.put_nowait((model, dev))
            self.devices.append(dev)
            print(f"  [Pool Worker] {dev} initialized in {time.time() - t0:.2f}s")

        self._initialized = True
        print(f"[IndicF5 Dual-GPU Pool] All workers ready in {time.time() - t0_all:.2f}s")

    async def get_worker(self):
        if not self._initialized:
            self.init_workers()
        return await self.queue.get()

    def release_worker(self, worker):
        self.queue.put_nowait(worker)

worker_pool = DualGPUWorkerPool()

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
    vram_stats = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / (1024**3)
            vram_stats.append(f"GPU {i}: {used:.2f} GiB")
    return {
        "status": "ok",
        "engine": "IndicF5 (Dual-GPU Parallel Worker Pool)",
        "concurrency": "Dual-GPU Hardware Parallel (GPU 0 + GPU 1)",
        "devices": worker_pool.devices,
        "available_workers": worker_pool.queue.qsize(),
        "vram": vram_stats,
        "default_voice": "Shubh (Bulbul v3 Persona)",
        "sample_rate": 24000,
        "features": [
            "dual-gpu-hardware-concurrency",
            "zero-shot-voice-cloning",
            "duration-patched-mode-a",
            "sentence-chunking-hallucination-guard",
            "fast-ode-18-steps",
            "indicxlit-unified-devanagari",
            "openai-speech-v1"
        ]
    }

def split_into_sentence_chunks(text: str, max_chars: int = 95) -> list[str]:
    chunks = []
    current = ""
    sentences = re.split(r"(?<=[;:,.!?।\n])\s+|(?<=[；：，।。！？\n])", text)
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) <= max_chars:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks if chunks else [text]

def cross_fade_waves(waves, sr=24000, dur=0.15):
    if not waves:
        return np.array([], dtype=np.float32)
    if len(waves) == 1:
        return waves[0]
    final = waves[0]
    k = int(dur * sr)
    for i in range(1, len(waves)):
        p, n = final, waves[i]
        samples = min(k, len(p), len(n))
        if samples <= 0:
            final = np.concatenate([p, n])
            continue
        fo = np.linspace(1, 0, samples, dtype=np.float32)
        fi = np.linspace(0, 1, samples, dtype=np.float32)
        overlap = p[-samples:] * fo + n[:samples] * fi
        final = np.concatenate([p[:-samples], overlap, n[samples:]])
    return final

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

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

        # 3. Model Inference with Dual-GPU Parallel Chunk Dispatch
        chunks = split_into_sentence_chunks(target_text, max_chars=95)
        t0 = time.time()
        used_devices = set()

        async def _synthesize_chunk(chunk_idx: int, chunk_text: str):
            worker, dev = await worker_pool.get_worker()
            used_devices.add(dev)
            try:
                def _infer():
                    with torch.inference_mode():
                        raw = worker(
                            text=chunk_text,
                            ref_audio_path=ref_audio_path,
                            ref_text=ref_text,
                        )
                    if hasattr(raw, "cpu"):
                        raw = raw.cpu().numpy()
                    arr = np.asarray(raw, dtype=np.float32).squeeze()
                    if raw.dtype == np.int16 or np.max(np.abs(arr)) > 1.5:
                        arr = arr.astype(np.float32) / 32768.0
                    return arr

                audio_chunk = await asyncio.to_thread(_infer)
                return (chunk_idx, audio_chunk)
            finally:
                worker_pool.release_worker((worker, dev))

        # Launch all chunks concurrently across both GPU workers
        chunk_tasks = [_synthesize_chunk(i, c) for i, c in enumerate(chunks) if c and c.strip()]
        results = await asyncio.gather(*chunk_tasks)
        results.sort(key=lambda x: x[0])
        waves = [r[1] for r in results]
        audio_arr = cross_fade_waves(waves, sr=24000, dur=0.15)
        latency = time.time() - t0

        # 4. Format Output Array
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
                "X-GPU-Devices": ",".join(sorted(used_devices)),
                "X-Chunks-Count": str(len(chunks)),
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

@app.get("/")
async def serve_studio():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    raise HTTPException(status_code=404, detail="Web Studio index.html not found")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="IndicF5 Production Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    # Pre-warm dual-GPU worker pool across all GPUs
    worker_pool.init_workers()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
