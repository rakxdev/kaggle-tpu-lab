#!/usr/bin/env python3
"""Phase 0b / TPU — step 3: load, generation proof, first benchmark.

Loads Qwen3.8-Flash-Next (W4A16) TP=8 on the v5e-8 via the fork's JAX path,
greets the world with a greedy proof, then measures decode and prefill.
First run pays full XLA compile (no cache yet) — expect a long load.
"""

import os
import time

os.environ.setdefault("TPU_BACKEND_TYPE", "jax")
os.environ["VLLM_XLA_CACHE_PATH"] = "/kaggle/working/xla_cache"
# The runtime venv (/tmp/venv, see venv_setup.py) ships its own libtpu; don't
# let the image's TPU_LIBRARY_PATH override it (serve_qwen38.py's rule).
os.environ.pop("TPU_LIBRARY_PATH", None)
CKPT = "/kaggle/tmp/ckpt"
CTX = 32768  # first-load context: small on purpose; 262k is a later step

import jax  # noqa: E402
print("devices:", jax.devices(), flush=True)
assert len(jax.devices()) == 8

# The checkpoint's model_type (qwen4_exp) is not in stock transformers; this
# fork call registers the HF config mapping and the JAX model impl for THIS
# process (setup_env.py also drops a .pth so spawned engine workers get it).
from tpu_inference.models.jax.qwen4_exp.startup import install  # noqa: E402
_info = install()
print("qwen4_exp install:", _info or "ok", flush=True)

from vllm import LLM, SamplingParams  # noqa: E402

t0 = time.time()
llm = LLM(model=CKPT, tensor_parallel_size=8, max_model_len=CTX,
          max_num_seqs=4, gpu_memory_utilization=0.85)
load_min = (time.time() - t0) / 60
print(f"\nLOAD+COMPILE: {load_min:.1f} min", flush=True)

# ---- generation proof ----
proof = llm.generate(["The capital of Germany is"],
                     SamplingParams(temperature=0, max_tokens=32))
text = proof[0].outputs[0].text
print(f"\nPROOF OUTPUT: {text[:200]!r}", flush=True)

# ---- decode benchmark ----
t0 = time.time()
out = llm.generate(
    ["Explain how mixture-of-experts models reduce inference cost."],
    SamplingParams(temperature=0, max_tokens=256))
dt = time.time() - t0
n = len(out[0].outputs[0].token_ids)
print(f"DECODE: {n} tokens in {dt:.1f}s = {n / dt:.1f} tok/s (incl. prefill)", flush=True)

# ---- prefill benchmark ----
tok = llm.get_tokenizer()
long_prompt = ("Context: " + ("The quick brown fox jumps over the lazy dog. " * 900)
               + "\nQuestion: which animal jumps?\nAnswer:")
n_prompt = len(tok.encode(long_prompt))
t0 = time.time()
llm.generate([long_prompt], SamplingParams(temperature=0, max_tokens=8))
dt = time.time() - t0
print(f"PREFILL: ~{n_prompt} tokens in {dt:.1f}s = {n_prompt / dt:.0f} tok/s", flush=True)

print("\nPHASE 0B COMPLETE — send all of this output back", flush=True)
