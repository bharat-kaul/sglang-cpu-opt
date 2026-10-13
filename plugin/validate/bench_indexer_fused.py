#!/usr/bin/env python3
"""Indexer-logits FUSED q-prep A/B (Stage 1.5 lever: the ~62% convert/pack/dispatch overhead, job 385397).
Fused (default) convert+transpose q directly into a reused bf16 buffer during pack; legacy
(INDEXER_NO_FUSED_QPACK=1) materializes qb=q.to(bf16) + per-n transpose().contiguous(). Same binary, env A/B.
Checks BIT-EXACT equality then measures streaming speedup at the M-sweep. >=3 processes (sbatch)."""
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
MS = [1, 8, 16, 32, 64]
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


def main():
    il = load(name="idx_fused", sources=[os.path.join(KD, "indexer_logits.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# identity HEAD={sha} proc={os.environ.get('PROC','0')} threads={torch.get_num_threads()} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} node={os.uname().nodename}")

    # ---- BIT-EXACT gate: fused vs legacy, every M ----
    print("[bit-exact] fused vs legacy (INDEXER_NO_FUSED_QPACK)")
    worst = 0.0
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        os.environ.pop("INDEXER_NO_FUSED_QPACK", None)
        a = il.indexer_logits(q, kv, w)
        os.environ["INDEXER_NO_FUSED_QPACK"] = "1"
        b = il.indexer_logits(q, kv, w)
        os.environ.pop("INDEXER_NO_FUSED_QPACK", None)
        md = float((a - b).abs().max())
        worst = max(worst, md)
        print(f"  M={M:>3} max|fused-legacy|={md:.3e} {'OK' if md == 0.0 else 'DIFF'}")
    print(f"  worst abs diff = {worst:.3e} -> {'BIT-EXACT' if worst == 0.0 else 'NOT bit-exact'}")

    # ---- streaming A/B ----
    print("\n[streaming] fused vs legacy (spd = legacy/fused, >1 => fused faster)")
    print(f"{'M':>4} {'legacy_us':>10} {'fused_us':>9} {'spd':>6}")
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        os.environ["INDEXER_NO_FUSED_QPACK"] = "1"
        t_leg = tstream(lambda: il.indexer_logits(q, kv, w))
        os.environ.pop("INDEXER_NO_FUSED_QPACK", None)
        t_fus = tstream(lambda: il.indexer_logits(q, kv, w))
        print(f"{M:>4} {t_leg:>10.1f} {t_fus:>9.1f} {t_leg/t_fus:>6.3f}")


if __name__ == "__main__":
    main()
