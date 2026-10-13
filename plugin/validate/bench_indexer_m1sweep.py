#!/usr/bin/env python3
"""Indexer-logits M=1 PARALLELISM sweep (test "overcome a dispatch bound by reducing parallelism"). The m1
path does one brgemm over all S + a parallel epilogue over S; INDEXER_M1_GRAIN sets the epilogue grain ->
effective thread count (grain=ceil(S/K) ~ K threads; grain>=S = serial). If T_full drops as threads fall, the
M=1 bound was over-parallelization (reduce it); if flat, the cost is elsewhere (alloc/brgemm) and the bound is
not in the epilogue parallelism. Bit-exact across grain (pure parallelization). Streaming+warm, >=3 procs."""
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
H, D, S = 64, 128, 1024
# (label, grain) -> approx effective threads
GRAINS = [("auto", 0), ("64t", 16), ("16t", 64), ("8t", 128), ("4t", 256), ("2t", 512), ("serial", 1024)]
torch.manual_seed(0)
from torch.utils.cpp_extension import load

_FLUSH = None; _SINK = 0.0


def _flush():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)
    _FLUSH.uniform_(-1.0, 1.0); _SINK += float(_FLUSH[::4096].sum())


def _call(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def tstream(fn, it=30):
    fn()
    return st.median([(_flush(), _call(fn))[1] for _ in range(it)]) * 1e6


def twarm(fn, it=60):
    fn()
    return st.median([_call(fn) for _ in range(it)]) * 1e6


def main():
    il = load(name="idx_m1", sources=[os.path.join(KD, "indexer_logits.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# identity HEAD={sha} proc={os.environ.get('PROC','0')} threads={torch.get_num_threads()} node={os.uname().nodename}")
    q = torch.randn(1, H, D); kv = torch.randn(1, S, D); w = torch.randn(1, H)

    os.environ.pop("INDEXER_M1_GRAIN", None); ref = il.indexer_logits(q, kv, w)
    worst = 0.0
    for _, g in GRAINS:
        os.environ["INDEXER_M1_GRAIN"] = str(g)
        worst = max(worst, float((il.indexer_logits(q, kv, w) - ref).abs().max()))
    os.environ.pop("INDEXER_M1_GRAIN", None)
    print(f"[bit-exact across grain] worst={worst:.3e} -> {'BIT-EXACT' if worst == 0 else 'DIFF'}")

    print("\n[M=1 epilogue parallelism sweep] T_full streaming us (warm in parens)")
    print(f"{'eff_threads':>12} {'grain':>6} {'stream_us':>10} {'warm_us':>9}")
    for lbl, g in GRAINS:
        os.environ["INDEXER_M1_GRAIN"] = str(g)
        s = tstream(lambda: il.indexer_logits(q, kv, w))
        wm = twarm(lambda: il.indexer_logits(q, kv, w))
        print(f"{lbl:>12} {g:>6} {s:>10.1f} {wm:>9.1f}")
    os.environ.pop("INDEXER_M1_GRAIN", None)


if __name__ == "__main__":
    main()
