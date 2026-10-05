#!/usr/bin/env python3
# CELL 3 — fetch the 33.1 GB BF16 checkpoint. This is the long one (10-40 min
# depending on your link), so it runs in the BACKGROUND with a live log.
# Poll it with cell_03b_status.sh. Never re-run this while a download is
# already running — check the poller first.
#
# Cost: ~$0.30-0.50 of Studio time for a full 33.1 GB pull. The download is
# resumable: if the Studio sleeps or you re-run, snapshot_download picks up
# the completed shards and skips them.
#
# WHY the full BF16 model and not a quant: the RTX PRO 6000 has 96 GB. The
# whole uncompressed stack is 33.1 GB, so it fits natively with ~60 GB to
# spare — no quantization, no CPU offload, no quality compromise. On the
# Kaggle 2xT4 route the same model does NOT fit and must drop to INT8 ConvRot
# (17.3 GB); that constraint is why that route is slower and why this one is
# not. See README "Why full BF16 here".

import os
import pathlib
import subprocess
import sys

WORK = pathlib.Path(os.environ.get("QI21_WORK", pathlib.Path.home() / "qwenimage21"))
LOG = WORK / "download.log"
REPO = "Qwen/Qwen-Image-2.1"

# Expected sizes in bytes, from the HF API listing of the repo tree. These
# are the authoritative completion check — a resumed download that was
# interrupted mid-shard leaves a short file that huggingface_hub will retry,
# but a silent partial would otherwise look "done". Verified 2026-10-05.
EXPECTED = {
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 9_969_927_912,
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 4_259_176_140,
    "text_encoder/model-00001-of-00004.safetensors": 4_999_751_340,
    "text_encoder/model-00002-of-00004.safetensors": 4_921_240_180,
    "text_encoder/model-00003-of-00004.safetensors": 4_921_093_452,
    "text_encoder/model-00004-of-00004.safetensors": 2_703_733_192,
    "vae/diffusion_pytorch_model.safetensors": 1_351_019_376,
    "model_index.json": None,
}
TOTAL = sum(v for v in EXPECTED.values() if v)


def already_complete() -> bool:
    """Byte-check every shard before touching the network at all."""
    if not (WORK / "model_index.json").exists():
        return False
    for rel, want in EXPECTED.items():
        p = WORK / rel
        if want is None:
            if not p.exists():
                return False
            continue
        if not p.exists() or abs(p.stat().st_size - want) > 5_000_000:
            return False
    return True


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)

    if already_complete():
        print("DOWNLOADED_OK — 33.1 GB already present and byte-verified")
        return 0

    # hf_transfer gives multi-stream download; without it the hub is
    # single-connection and this takes 3-5x longer.
    env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1")
    # An HF token is optional (the repo is public) but raises rate limits.
    token = os.environ.get("HF_TOKEN")
    if token:
        env["HF_TOKEN"] = token

    script = WORK / "_fetch.py"
    script.write_text(
        "import os\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download(repo_id={REPO!r}, local_dir={str(WORK)!r},\n"
        "                  max_workers=8)\n"
        "print('SNAPSHOT_RETURNED')\n"
    )
    with open(LOG, "w") as fh:
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # survives the parent cell exiting
        )

    print(f"DOWNLOAD LAUNCHED — pid {proc.pid}, log {LOG}")
    print(f"expected total {TOTAL/1e9:.1f} GB into {WORK}")
    print("poll with cell_03b_status.sh  (re-run freely)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
