#!/usr/bin/env python3
"""Indexer-logits TILE-COUNT SWEEP (test the dispatch-bound hypothesis). Vary the tiles-per-thread target
(INDEXER_TPT) -> total tiles = TPT*threads. If T_full DROPS as tiles shrink toward the core-count floor (TPT=1
=> ~64 tiles) the op is dispatch-bound and fewer tiles is the lever; if FLAT, the fixed cost is barriers/
thread-pool and tiling is NOT the fix (hypothesis refuted). Bit-exact across TPT (tiling doesn't change math).
Streaming + warm, >=3 processes (sbatch)."""
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
MS = [8, 16, 32, 64]
TPTS = [1, 2, 4, 8]
H, D, S = 64, 128, 1024
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


def tstream(fn, it=25):
    fn()
    return st.median([(_flush(), _call(fn))[1] for _ in range(it)]) * 1e6


def twarm(fn, it=40):
    fn()
    return st.median([_call(fn) for _ in range(it)]) * 1e6


def main():
    il = load(name="idx_tsw", sources=[os.path.join(KD, "indexer_logits.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# identity HEAD={sha} proc={os.environ.get('PROC','0')} threads={torch.get_num_threads()} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} node={os.uname().nodename}")

    # bit-exact across TPT (reference = default TPT=4)
    print("[bit-exact across TPT vs TPT=4]")
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        os.environ.pop("INDEXER_TPT", None); ref = il.indexer_logits(q, kv, w)
        worst = 0.0
        for t in TPTS:
            os.environ["INDEXER_TPT"] = str(t)
            worst = max(worst, float((il.indexer_logits(q, kv, w) - ref).abs().max()))
        os.environ.pop("INDEXER_TPT", None)
        print(f"  M={M:>3} worst|TPT-ref|={worst:.3e} {'OK' if worst == 0 else 'DIFF'}")

    # sweep
    print("\n[tile sweep] T_full streaming us (warm in parens); tiles = TPT*threads")
    hdr = "   M  " + "".join(f"TPT={t:<10}" for t in TPTS)
    print(hdr)
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        cells = []
        for t in TPTS:
            os.environ["INDEXER_TPT"] = str(t)
            s = tstream(lambda: il.indexer_logits(q, kv, w))
            wmt = twarm(lambda: il.indexer_logits(q, kv, w))
            cells.append(f"{s:6.1f}({wmt:5.1f})")
        os.environ.pop("INDEXER_TPT", None)
        print(f"  {M:>3}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
