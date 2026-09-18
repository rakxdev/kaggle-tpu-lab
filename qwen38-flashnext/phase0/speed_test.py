#!/usr/bin/env python3
"""Download-bandwidth probe: 1 stream vs 8 parallel streams from the HF CDN.

Measures aggregate throughput for 256 MiB ranges of a real checkpoint shard.
The running packer shares the pipe, so both measurements are taken under the
same conditions — the 8x/1x RATIO is what matters, not the absolute numbers.
"""

import concurrent.futures as cf
import time
import urllib.request

URL = ("https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16"
       "/resolve/main/model-00002.safetensors")
CH = 1 << 28  # 256 MiB per stream


def grab(index):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={index * CH}-{(index + 1) * CH - 1}"})
    t0, got = time.time(), 0
    with urllib.request.urlopen(req, timeout=180) as r:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            got += len(b)
    return got, time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    got, dt = grab(20)
    print(f"1-stream : {got / 1e9:.2f} GB in {dt:5.1f}s  = {got / 1e6 / dt:5.0f} MB/s",
          flush=True)

    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda i: grab(40 + i), range(8)))
    got = sum(g for g, _ in res)
    dt = time.time() - t0
    print(f"8-stream : {got / 1e9:.2f} GB in {dt:5.1f}s  = {got / 1e6 / dt:5.0f} MB/s aggregate",
          flush=True)
