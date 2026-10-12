# Index-Translate-9B Multimodal (Text + Vision) on Kaggle T4 ×2

**Serve Index-Translate-9B Q4_K_M with its companion multimodal vision encoder (`mmproj`) on Kaggle's free 2× T4 GPUs with a public Cloudflare tunnel, API-key authentication, and native support for Immersive Translate on Firefox (Linux) and Kiwi Browser (Android).**

- **Model**: [`IndexTeam/Index-Translate-9B-GGUF`](https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF)
- **Architecture**: Qwen3.5 hybrid DeltaNet + linear attention (150 languages supported).
- **Vision Support**: Powered by `Index-Translate-9B.mmproj-Q8_0.gguf` (zero OCR dependency; translates diagrams, comics, infographics, and screenshots directly).
- **Format**: Standard OpenAI-compatible `/v1/chat/completions` supporting both text and base64 `image_url`.

---

## Hardware & Speed Profile (Kaggle 2× T4, 31.6 GiB VRAM total)

| Component | VRAM Used | Notes |
|---|---|---|
| **Text Model (`Index-Translate-9B.Q4_K_M.gguf`)** | **5.78 GB** | Medium 4-bit k-quant (99.2% FP16 quality) |
| **Vision Encoder (`Index-Translate-9B.mmproj-Q8_0.gguf`)** | **0.62 GB** | High-precision Q8_0 multimodal vision projector |
| **KV Cache & Activations (40k context, 10 slots)** | **~1.5 GB** | Leaves **~7.9 GB free headroom** on GPU 0 |
| **Total VRAM on GPU 0** | **~7.9 GB** (out of 15.8 GB) | **100% on GPU 0 — Zero PCIe split latency!** |
| **GPU 1** | **0 GB** (idle) | Can run second worker or stay cool |
| **Parallel Concurrency** | **10 simultaneous slots** (`-np 10`) | Translates 10 paragraphs/images at once |
| **Total Aggregate Throughput** | **~100–120 tok/s** | ~200 ms per text paragraph |
| **Image Translation Latency** | **~1.2 – 1.8 s** | Translates web images and comics as you scroll |

---

## Notebook Cells (Run in Order)

In a Kaggle Notebook with **GPU T4 ×2** and **Internet ON**:

### 1. Setup Dependencies
```python
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/index-translate-kaggle/cell_00_deps.sh | sh
```

### 2. Pull Kit
```python
!curl -sSL https://raw.githubusercontent.com/rakxdev/kaggle-tpu-lab/main/index-translate-kaggle/cell_00b_pull.sh | sh
```

### 3. Hardware Report
```python
!sh /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_01_report.sh
```

### 4. Setup llama-server (Fast Prebuilt or Ninja build)
```python
!sh /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_02_setup.sh
```
*(If building from source, poll with: `!sh /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_02b_status.sh`)*

### 5. Download Model & Vision Encoder (6.4 GB total, ~45s)
```python
!python3 /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_03_download.py
```
*(Poll and verify with: `!sh /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_03b_status.sh` until `SIZES_OK`)*

### 6. Smoke Test (Verifies text translation + image translation)
```python
!python3 /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_04_smoke.py
```
*(Prints `INDEX_SMOKE_OK`)*

### 7. Launch Public Endpoint (Cloudflare Tunnel + Watchdog)
```python
!sh /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_05_serve.sh
```
Prints your public URL:
```text
https://<random-name>.trycloudflare.com
API Key: index-...
```

### 8. End-to-End Proof (Optional, run from notebook or laptop)
```python
!python3 /kaggle/working/kaggle-tpu-lab/index-translate-kaggle/cell_05b_proof.py <YOUR-URL> <YOUR-KEY>
```

---

## Immersive Translate Browser Configuration

### For Firefox (Linux Desktop) & Kiwi Browser (Android Mobile):

1. Open **Immersive Translate Options** $\rightarrow$ **Translation Services** $\rightarrow$ **Custom AI** (or OpenAI).
2. Configure:
   - **Custom API interface address**:
     ```
     https://<your-subdomain>.trycloudflare.com/v1/chat/completions
     ```
   - **APIKEY**:
     ```
     <your-api-key-printed-in-cell-5>
     ```
   - **Model**:
     Check ☑ **Enter the name of the custom model** and enter:
     ```
     Index-Translate-9B
     ```
3. Enable **Comics / Images Translation**:
   - In the left sidebar of Immersive Translate, click **Comics/Images**.
   - Select your custom translation service as the engine.
   - Now hovering over or translating any webpage image, chart, or comic panel automatically calls the vision encoder (`mmproj`) and translates the text!
4. Click **Verify service** $\rightarrow$ Green checkmark `✓`!
