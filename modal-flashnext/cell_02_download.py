# CELL 2 — write the downloader, then run it on Modal. One cell, progress streams
# directly below. Pulls ~87 GB into the Volume (IQ3_S 83.6 GB + mmproj 0.91 GB +
# MTP head ~2 GB). CPU-only, costs cents, usually 5-15 min. Re-running the cell
# resumes/skips files already on the Volume.

import pathlib
import subprocess

DOWNLOADER = '''
import glob
import subprocess

import modal

# Optional: paste your HF token to lift rate limits. Both repos are public.
HF_TOKEN = ""  # stays in this generated file inside the notebook session only

vol = modal.Volume.from_name("flashnext-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", **({"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {})})
)

app = modal.App("flashnext-download", image=image)

MAIN = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"   # IQ3_S/ dir + shared mmproj
MTP = "quimmedes/Qwen3.8-Flash-Next-MTP-GGUF"          # MTP draft head for llama.cpp


@app.function(volumes={"/cache": vol}, timeout=4 * 3600, cpu=4, memory=16384)
def pull():
    from huggingface_hub import snapshot_download

    # 2 shards: 54.8 GB weights + 28.8 GB n-gram table, plus the vision projector
    snapshot_download(
        MAIN,
        local_dir="/cache/gguf",
        allow_patterns=["IQ3_S/*", "mmproj-Qwen3.8-Flash-Next-BF16.gguf"],
    )
    # pattern must match only the MTP head, not the repo's UD-IQ3_XXS main shards
    snapshot_download(MTP, local_dir="/cache/gguf/mtp", allow_patterns=["*Q4_K_M*"])

    mtp = glob.glob("/cache/gguf/mtp/**/*.gguf", recursive=True)
    assert mtp, "MTP head missing - check quimmedes repo file names"
    vol.commit()

    print("DOWNLOADED_OK")
    print(subprocess.run(["du", "-sh", "/cache/gguf"], capture_output=True, text=True).stdout)
    print(subprocess.run(
        ["find", "/cache/gguf", "-name", "*.gguf", "-printf", "%s %p\\n"],
        capture_output=True, text=True).stdout)


@app.local_entrypoint()
def main():
    pull.remote()
'''

pathlib.Path("download_weights.py").write_text(DOWNLOADER)
compile(DOWNLOADER, "download_weights.py", "exec")  # syntax gate before sending to Modal
print("wrote download_weights.py - running (progress streams below)")
subprocess.run(["modal", "run", "download_weights.py"], check=True)
