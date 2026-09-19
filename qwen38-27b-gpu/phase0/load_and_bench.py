#!/usr/bin/env python3
"""qwen38-27b-gpu / phase0 — step 3: load, generation proof, benchmark.

Offline vLLM path, TP=2 across both T4s, W4A16 auto-detected from the
compressed-tensors config, MTP speculative decoding enabled (single stream —
the known GDN+MTP batch>=4 acceptance bug does not apply at batch 1).
First load compiles; keep context at 32k to bound that cost.
"""

import os
import time

CKPT = "/kaggle/tmp/ckpt"
CTX = 32768

os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

from vllm import LLM, SamplingParams  # noqa: E402

t0 = time.time()
llm = LLM(
    model=CKPT,
    tensor_parallel_size=2,
    max_model_len=CTX,
    max_num_seqs=4,
    gpu_memory_utilization=0.92,
    speculative_config={"method": "mtp", "num_speculative_tokens": 3},
)
print(f"\nLOAD+COMPILE: {(time.time() - t0) / 60:.1f} min", flush=True)

proof = llm.generate(["The capital of Germany is"],
                     SamplingParams(temperature=0, max_tokens=32))
print(f"\nPROOF OUTPUT: {proof[0].outputs[0].text[:200]!r}", flush=True)

t0 = time.time()
out = llm.generate(
    ["Explain how mixture-of-experts models reduce inference cost."],
    SamplingParams(temperature=0, max_tokens=256))
dt = time.time() - t0
n = len(out[0].outputs[0].token_ids)
print(f"DECODE: {n} tokens in {dt:.1f}s = {n / dt:.1f} tok/s (incl. prefill)", flush=True)

tok = llm.get_tokenizer()
long_prompt = ("Context: " + ("The quick brown fox jumps over the lazy dog. " * 900)
               + "\nQuestion: which animal jumps?\nAnswer:")
n_prompt = len(tok.encode(long_prompt))
t0 = time.time()
llm.generate([long_prompt], SamplingParams(temperature=0, max_tokens=8))
dt = time.time() - t0
print(f"PREFILL: ~{n_prompt} tokens in {dt:.1f}s = {n_prompt / dt:.0f} tok/s", flush=True)

print("\nPHASE 0B-GPU COMPLETE — send all of this output back", flush=True)
