#!/usr/bin/env python3
# CELL 4 — smoke test: load the full 33.1 GB pipeline and make ONE 1024x1024
# image. This is the cell that tells you whether the stack works at all, and
# it is deliberately the cheapest possible proof: one image, one resolution,
# the official 40-step setting, no benchmark loop.
#
# Cost: the load moves 33.1 GB from disk into VRAM (a minute or two on NVMe)
# and the image takes whatever the card actually does — budget ~$0.20.
#
# Run this BEFORE cell 5. If it fails you want to know on a $0.20 cell, not
# after a benchmark loop has burned an hour.
#
# NOTE ON DTYPE: the whole point of this route is that 33.1 GB fits in 96 GB,
# so we load straight to bf16 with no quantization and no CPU offload. Do NOT
# add pipe.enable_model_cpu_offload() here — it would reintroduce the PCIe
# shuffling that makes the Kaggle 2xT4 route slow, for no benefit on a 96 GB
# card. And do not "optimise" bfloat16 to float16: bf16 has fp32's exponent
# range, fp16 does not, and this model's residual streams reach ~3.4e8, which
# overflows fp16 to NaN and yields black images. QwenImage21Pipeline only
# offers [bfloat16, float32] anyway.

import os
import pathlib
import time

import torch

WORK = pathlib.Path(os.environ.get("QI21_WORK", pathlib.Path.home() / "qwenimage21"))
OUT = WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, "
    "reflections on wet pavement"
)


def main() -> int:
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        print("!! no CUDA — the Studio has no GPU attached. Paste this back.")
        return 1

    t0 = time.time()
    print(f"loading pipeline from {WORK} ... (33.1 GB, this is the slow part)")
    pipe = QwenImage21Pipeline.from_pretrained(
        str(WORK),
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    load_s = time.time() - t0

    vram = torch.cuda.memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"LOADED in {load_s:.1f}s — {vram:.1f} GiB resident of {total:.1f} GiB")
    if vram > total * 0.9:
        print("!! weights nearly fill the card — raise the risk of OOM at 2K")

    # The official card's example: 40 steps, CFG 1, Euler/simple, 1024x1024.
    # CFG 1 is the documented default for this model — it is NOT a mistake.
    t1 = time.time()
    image = pipe(
        prompt=PROMPT,
        width=1024,
        height=1024,
        num_inference_steps=40,
        true_cfg_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    gen_s = time.time() - t1

    path = OUT / "smoke_1024_s42.png"
    image.save(path)

    # A black or NaN render is the classic failure on older cards. Check the
    # pixel statistics rather than trusting that a PNG was written.
    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    std = float(arr.std())
    mean = float(arr.mean())
    peak = torch.cuda.max_memory_allocated() / 2**30

    print(f"IMAGE {path}  {image.size[0]}x{image.size[1]}")
    print(f"GENERATE {gen_s:.1f}s at 40 steps = {gen_s/40:.2f} s/step")
    print(f"VRAM peak {peak:.1f} GiB")
    print(f"pixels mean={mean:.1f} std={std:.1f}")

    if std < 1.0:
        print("!! image is essentially flat/black — that is the NaN signature.")
        print("!! paste this output; do not run cell 5.")
        return 1

    print("SMOKE_OK — full BF16 pipeline works on this card")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
