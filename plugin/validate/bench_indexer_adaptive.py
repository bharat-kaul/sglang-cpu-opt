#!/usr/bin/env python3
"""Indexer ISA-adaptive test: does an AVX-512 (fp32 at::bmm) score path beat the AMX brgemm at the OP level,
and is it CONFORMANT (top-k set-match)? Decides the adaptive AMX<->AVX dispatch crossover for the indexer.

Per M: (1) op-level streaming latency of shipped indexer_logits (AMX brgemm) vs indexer_logits_fp32avx (AVX-512);
(2) top-k SET-MATCH of the two logit outputs (the indexer feeds top-k selection, so the gate is set membership,
not bit-exactness). k sweeps a realistic range. EMR (AMX + avx512_bf16).
"""
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
MS = [1, 2, 4, 8, 16, 32, 64]
KS = [64, 256, 512]                                   # realistic top-k selection sizes
torch.manual_seed(0)
from torch.utils.cpp_extension import load

_FLUSH = None
_SINK = 0.0


def _flush():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)
    _FLUSH.uniform_(-1.0, 1.0)
    _SINK += float(_FLUSH[::4096].sum())


def _call(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def tstream(fn, it=25):
    fn()
    return st.median([(_flush(), _call(fn))[1] for _ in range(it)]) * 1e6


def _setmatch(a, b, k):
    # fraction of rows whose top-k INDEX SET matches between the two logit tensors
    ia = torch.topk(a, k, dim=1, sorted=False).indices
    ib = torch.topk(b, k, dim=1, sorted=False).indices
    m = 0
    for r in range(a.size(0)):
        if set(ia[r].tolist()) == set(ib[r].tolist()):
            m += 1
    return m / a.size(0)


def main():
    il = load(name="adap_il", sources=[os.path.join(KD, "indexer_logits.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# indexer ISA-adaptive HEAD={sha} threads={torch.get_num_threads()} node={os.uname().nodename}")
    print(f"{'M':>4} {'AMX_us':>8} {'AVX_us':>8} {'AVX_speedup':>11} | setmatch@k " + " ".join(f"k{k}" for k in KS))
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        l_amx = il.indexer_logits(q, kv, w)
        l_avx = il.indexer_logits_fp32avx(q, kv, w)
        t_amx = tstream(lambda: il.indexer_logits(q, kv, w))
        t_avx = tstream(lambda: il.indexer_logits_fp32avx(q, kv, w))
        sm = [f"{_setmatch(l_amx, l_avx, k):.3f}" for k in KS]
        print(f"{M:>4} {t_amx:>8.1f} {t_avx:>8.1f} {t_amx / t_avx:>10.2f}x | {'':>10} " + "  ".join(sm))
    print("# AVX_speedup>1 => AVX-512 fp32 score path wins the OP at that M. setmatch=1.000 => top-k selection")
    print("# is identical (conformant); <1 => fp32 scores flip some selections (conformance decision).")


if __name__ == "__main__":
    main()
