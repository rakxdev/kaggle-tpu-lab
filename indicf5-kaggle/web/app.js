// IndicF5 Master Audio Studio — Client Logic

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
const metricGpu = document.getElementById('metricGpu');
const chkNormalize = document.getElementById('chkNormalize');
const canvas = document.getElementById('visualizerCanvas');
const canvasCtx = canvas.getContext('2d');
const visIdleText = document.getElementById('visIdleText');

// Voice selection
const cardShubh = document.getElementById('cardShubh');
const cardCustom = document.getElementById('cardCustom');
const customVoiceSection = document.getElementById('customVoiceSection');
const customAudioFile = document.getElementById('customAudioFile');
const customAudioText = document.getElementById('customAudioText');

let currentVoice = 'shubh';
let currentAudioUrl = null;
let customBase64Audio = null;

// Voice switching
cardShubh.addEventListener('click', () => {
  currentVoice = 'shubh';
  cardShubh.classList.add('active');
  cardCustom.classList.remove('active');
  customVoiceSection.style.display = 'none';
});

cardCustom.addEventListener('click', () => {
  currentVoice = 'custom';
  cardCustom.classList.add('active');
  cardShubh.classList.remove('active');
  customVoiceSection.style.display = 'block';
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

// Presets
document.querySelectorAll('.preset-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    textInput.value = chip.dataset.text;
    updatePreview();
  });
});

// Speed slider
speedSlider.addEventListener('input', (e) => {
  speedVal.innerText = parseFloat(e.target.value).toFixed(2) + 'x';
});

// Reference sample toggle
btnListenRef.addEventListener('click', () => {
  if (refAudioPlayer.paused) {
    refAudioPlayer.play();
    btnListenRef.innerText = '⏸ Pause Sample';
  } else {
    refAudioPlayer.pause();
    btnListenRef.innerText = '▶ Listen Sample';
  }
});
refAudioPlayer.onended = () => {
  btnListenRef.innerText = '▶ Listen Sample';
};

// Live preview debounce
let previewTimer = null;
function updatePreview() {
  const val = textInput.value;
  charCount.innerText = val.length + ' characters';
  clearTimeout(previewTimer);
  if (!val.trim()) {
    normalizedPreview.innerText = "Start typing to preview transliteration...";
    return;
  }
  previewTimer = setTimeout(async () => {
    try {
      const res = await fetch('/api/preview_normalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: val })
      });
      if (res.ok) {
        const data = await res.json();
        normalizedPreview.innerText = data.normalized;
      }
    } catch (err) {}
  }, 300);
}
textInput.addEventListener('input', updatePreview);

// Spectrum visualizer setup
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
  visIdleText.style.display = 'none';
  if (audioCtx.state === 'suspended') audioCtx.resume();
  drawVisualizer();
};
audioPlayer.onpause = () => {
  cancelAnimationFrame(animationId);
};
audioPlayer.onended = () => {
  cancelAnimationFrame(animationId);
  visIdleText.style.display = 'block';
};

// Safe Synthesis Handler (reads body stream once safely)
async function doSynthesize() {
  const text = textInput.value.trim();
  if (!text) {
    alert("Please enter text to synthesize.");
    return;
  }

  btnSynthesize.disabled = true;
  synthSpinner.style.display = 'inline-block';
  btnSynthText.innerText = "Synthesizing across Dual GPUs...";

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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      // Read response text ONCE to avoid "Body has already been consumed" error
      const txt = await res.text();
      let errDetail = "Request failed (HTTP " + res.status + ")";
      try {
        const errJson = JSON.parse(txt);
        if (errJson && errJson.detail) errDetail = errJson.detail;
      } catch (e) {
        if (txt) {
          errDetail = "Gateway/Server Error: " + txt.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim().slice(0, 150);
        }
      }
      throw new Error(errDetail);
    }

    const blob = await res.blob();
    const latency = (performance.now() - t0) / 1000;
    const durHeader = res.headers.get('X-Audio-Duration');
    const gpuHeader = res.headers.get('X-GPU-Devices') || res.headers.get('X-GPU-Device') || 'cuda';
    const duration = durHeader ? parseFloat(durHeader) : 0;
    const rtf = duration > 0 ? (latency / duration).toFixed(3) : '--';

    if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = URL.createObjectURL(blob);
    audioPlayer.src = currentAudioUrl;
    audioPlayer.play();

    metricDuration.innerText = duration.toFixed(2) + 's';
    metricLatency.innerText = 'Latency: ' + latency.toFixed(2) + 's';
    metricRtf.innerText = 'RTF: ' + rtf + 'x';
    if (metricGpu) metricGpu.innerText = 'GPU: ' + gpuHeader;

    btnDownload.style.display = 'inline-block';
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
    synthSpinner.style.display = 'none';
    btnSynthText.innerText = "Generate Cloned Speech (Ctrl + Enter)";
  }
}

btnSynthesize.addEventListener('click', doSynthesize);
window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    doSynthesize();
  }
});

// Initialization
window.addEventListener('load', () => {
  textInput.value = "बारिश की वो शाम कुछ अलग ही थी। पुरानी दिल्ली की उस छोटी सी नुक्कड़ वाली चाय की दुकान पर मैं चुपचाप बैठा था।";
  updatePreview();
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
});
