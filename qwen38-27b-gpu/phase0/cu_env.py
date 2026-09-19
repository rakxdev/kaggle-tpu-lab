#!/usr/bin/env python3
"""Kaggle's T4 image is a CUDA 12.8 host, but PyPI vllm 0.28.0 is a CUDA-13
build. Its compiled extension carries no rpath, and torch 2.13's cu130 wheels
install the CUDA-13 runtime into site-packages/nvidia/cu13/lib/ — a directory
the dynamic loader never searches on that image. `import vllm` therefore dies
with "libcudart.so.13: cannot open shared object file".

The fix is to put that directory on LD_LIBRARY_PATH. The loader only reads it
at exec time, so the fix is a re-exec of this process, not an os.environ
assignment. Import this module and call reexec() before anything imports vllm;
it is a no-op when the libs are already reachable (a normal CUDA-13 host, a
venv where the layout happens to resolve), so it is safe everywhere.
"""

import glob
import os
import sys

MARKER = "libcudart.so.13"


def cu13_dirs():
    """Every site-packages/nvidia/cu13/lib that actually holds the CUDA-13 runtime."""
    out, seen = [], set()
    for entry in sys.path:
        if not entry or not os.path.isdir(entry):
            continue
        for d in sorted(glob.glob(os.path.join(entry, "nvidia", "cu13", "lib"))):
            if d not in seen and os.path.exists(os.path.join(d, MARKER)):
                seen.add(d)
                out.append(d)
    return out


def needs_fix():
    dirs = cu13_dirs()
    if not dirs:
        return None
    have = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [d for d in dirs if d not in have]
    return missing or None


def reexec():
    """Re-exec this process with the CUDA-13 lib dirs on LD_LIBRARY_PATH."""
    missing = needs_fix()
    if not missing:
        return False
    cur = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + cur)
    print(f"[cu_env] {len(missing)} CUDA-13 lib dir(s) were off the loader path;"
          f" re-executing with them on LD_LIBRARY_PATH", flush=True)
    # argv[0] is replayed as given; cwd is unchanged across execve, so a
    # relative script path still resolves.
    os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)
    raise SystemExit("execve returned unexpectedly")


if __name__ == "__main__":
    reexec()
    print("cu13 dirs:", cu13_dirs(), "| on path:",
          [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if "cu13" in p])
