#!/usr/bin/env python3
# CELL 5 — benchmark. Measures the curve rather than asserting a number, so
# the results are YOUR card's numbers and stay true even if this file ages.
#
# Cost: this is the expensive cell. It loads the pipeline once (33.1 GB) and
# then runs a sweep. At ~$3.26/h budget roughly $1-2, i.e. 20-40 min. Set
# QUICK=1 for a fast pass (see below) if you just want the headline number.
#
#   QUICK=1 python3 cell_05_bench.py     # 1 arm only, ~$0.20
#   python3 cell_05_bench.py              # full sweep
#
# What it measures, and why each matters for serving:
#   1. cold vs warm  — the first image pays for kernel autotuning and memory
#      allocator growth. Throughput planning must use the warm number.
#   2. s/step         — the constant that tells you where your credit goes.
#      Everything else is steps x s/step.
#   3. steps sweep    — is 40 steps actually needed on 2.1, or is the curve
#      flat after 20? This is your biggest cost lever and nobody should take
#      it on faith.
#   4. resolution     — cost scales with the token count, so this shows the
#      real price of going to the model's native 2K.
#   5. torch.compile  — OPT-IN, off by default, and KNOWN BROKEN upstream.
#      diffusers issue #14821 ("Qwen Image 2.1 transformer is incompatible with
#      torch.compile", opened 2026-09-20, still open) reports four host-
#      dependent graph breaks, two of them data-dependent per step: use_kv_cache
#      makes step 0 run "extract" and later steps "cached", and build_token_metadata
#      uses nonzero() so index-vector LENGTH depends on mask values. That is
#      the recompile trigger. So COMPILE=1 is a curiosity, not a speedup —
#      fullgraph=True hard-errors, and plain compile may still work via graph
#      breaks (unconfirmed). The 2.1 docs page does recommend
#      QwenImage21FlexAttnProcessor "once the model is compiled"; we do not
#      use it, since the compile it depends on is not working yet.
#
# Every image is written to disk with its seed so you can look at them and
# judge quality at each step count yourself. s/step is a number; whether 20
# steps looks acceptable is a judgement only you can make.

import os
import pathlib
import statistics
import time

import torch

WORK = pathlib.Path(os.environ.get("QI21_WORK", pathlib.Path.home() / "qwenimage21"))
OUT = WORK / "bench"
OUT.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("QUICK") == "1"
COMPILE = os.environ.get("COMPILE") == "1"

PROMPT = (
    "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, "
    "reflections on wet pavement"
)
# Same prompt and seed across every arm, so any visual difference is
# attributable to step count alone and not to luck of the draw.
SEED = 42


def render(pipe, steps: int, w: int, h: int, tag: str) -> float:
    t0 = time.time()
    img = pipe(
        prompt=PROMPT,
        width=w,
        height=h,
        num_inference_steps=steps,
        true_cfg_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(SEED),
        # Pinned for the whole sweep. The docs warn that toggling this
        # does not reproduce the same image bit-for-bit in reduced
        # precision — a 1-ULP difference at block 1 is amplified through
        # 32 blocks and every step. Left at the default it would drift;
        # explicitly set, every arm is comparable AND reproducible.
        use_kv_cache=True,
    ).images[0]
    dt = time.time() - t0
    img.save(OUT / f"{tag}.png")
    return dt


def main() -> int:
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        print("!! no CUDA — stop and paste this output")
        return 1

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(WORK), torch_dtype=torch.bfloat16
    ).to("cuda")
    print(f"load {time.time()-t0:.1f}s — "
          f"{torch.cuda.memory_allocated()/2**30:.1f} GiB resident")

    results: list[tuple[str, int, int, int, float, float]] = []

    if COMPILE:
        # Deliberately after the plain numbers. Expect this to fail: see the
        # header and diffusers issue #14821. It is here so you can confirm the
        # bug still reproduces on your torch build, and so that if a fix lands
        # you have a before/after. A failure is a legitimate result.
        print("\n=== torch.compile (opt-in; known broken, see diffusers #14821) ===")
        try:
            pipe.transformer = torch.compile(
                pipe.transformer, mode="max-autotune", fullgraph=False
            )
            dt = render(pipe, 40, 1024, 1024, "compiled_40_1024")
            print(f"compiled 40 steps @1024: {dt:.1f}s = {dt/40:.2f} s/step")
            results.append(("compiled", 40, 1024, 1024, dt, dt / 40))
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"!! torch.compile failed ({type(exc).__name__}: {exc})")
            print("!! this is the EXPECTED outcome per diffusers #14821")
            print("!! re-assign the eager module so the sweeps below are valid:")
            pipe.transformer = pipe.transformer._orig_mod
            results.append(("compiled", 40, 1024, 1024, float("nan"), float("nan")))

    print("\n=== cold vs warm (40 steps, 1024x1024) ===")
    cold = render(pipe, 40, 1024, 1024, "cold_40_1024")
    print(f"cold: {cold:.1f}s = {cold/40:.2f} s/step")
    warm_runs = [render(pipe, 40, 1024, 1024, f"warm{i}_40_1024") for i in range(2)]
    warm = statistics.median(warm_runs)
    print(f"warm: {warm:.1f}s = {warm/40:.2f} s/step "
          f"(runs: {', '.join(f'{r:.1f}' for r in warm_runs)})")
    print(f"cold overhead: {cold-warm:.1f}s — plan capacity on the warm number")
    results.append(("cold", 40, 1024, 1024, cold, cold / 40))
    results.append(("warm", 40, 1024, 1024, warm, warm / 40))

    if QUICK:
        print("\nQUICK=1 — skipping the sweeps. Full sweep for the curves.")
    else:
        print("\n=== step-count sweep (1024x1024) ===")
        # Why 20/28/40 and NOT 4/8: low step counts are only meaningful with a
        # DISTILLED LoRA, and the base model at 4-8 steps is simply bad. There
        # is no LightX2V distilled variant for 2.1 (their org is all 1.x /
        # 2512 / Edit-2511). Real community 2.1 adapters exist — Turbo8 (8
        # step), Viggle turbo (6 step), ThakiCloud FewStep (5/8), PrunaAI
        # (5/8) — but every one of them requires scheduler surgery
        # (shift_terminal=None) to work at all, so testing them is a separate
        # exercise, not a line in this sweep. Do not read the 4/8 rows of some
        # other benchmark as equivalent to what you see here.
        #
        # Also note there is NO published guidance on exactly 20 steps for 2.1.
        # Nearest anchors: ComfyUI ships 25 as its default; hands/fine detail
        # "settles by about 30 steps"; 25->40 "reduces fizzle in detailed
        # areas". So 20 sits below the recommended floor — that is precisely
        # why this sweep exists, and why you look at the PNGs.
        for steps in (20, 28, 40):
            dt = render(pipe, steps, 1024, 1024, f"sweep_{steps}_1024")
            results.append(("sweep", steps, 1024, 1024, dt, dt / steps))
            print(f"  {steps:>2} steps: {dt:6.1f}s = {dt/steps:.2f} s/step")

        print("\n=== resolution sweep (20 steps) ===")
        # Cost tracks token count, and token count tracks pixels, so this is
        # how you price the jump to the model's native 2K output.
        for w, h in ((512, 512), (768, 768), (1328, 1328), (2048, 2048)):
            dt = render(pipe, 20, w, h, f"res_{w}x{h}")
            results.append(("res", 20, w, h, dt, dt / 20))
            print(f"  {w}x{h}: {dt:6.1f}s = {dt/20:.2f} s/step")
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"      VRAM peak {peak:.1f} GiB of "
                  f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f}")

    print("\n=== summary ===")
    print(f"{'arm':<10} {'steps':>5} {'res':>10} {'sec':>8} {'s/step':>8}")
    for arm, steps, w, h, dt, sps in results:
        if dt != dt:  # NaN — the expected torch.compile failure above
            print(f"{arm:<10} {steps:>5} {f'{w}x{h}':>10} {'FAILED':>8} {'-':>8}")
            continue
        print(f"{arm:<10} {steps:>5} {f'{w}x{h}':>10} {dt:>8.1f} {sps:>8.2f}")

    total_vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nVRAM peak overall: {total_vram:.1f} GiB")
    print(f"images in {OUT}")
    print("\nLook at the sweep images before picking a step count:")
    print("  s/step is arithmetic; whether 20 steps is good enough is your eye.")
    print()
    print("  CAVEAT on the demo prompt: it is the official model card's neon")
    print("  SIGN prompt, i.e. it is a text-rendering test. Text is the first")
    print("  thing to break at low step counts with no CFG — Turbo8's own eval")
    print("  scores text exact-match 75% vs its 40-step teacher's 95%. So the")
    print("  20-step arm will look worse than it is ON TEXT specifically. Do")
    print("  not conclude '20 steps is unusable' from a prompt whose whole")
    print("  point is legible lettering; re-test with a scene prompt too.")
    print("BENCH_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
