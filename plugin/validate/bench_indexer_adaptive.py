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
    print(f"# indexer ISA/structure HEAD={sha} threads={torch.get_num_threads()} node={os.uname().nodename}")
    print(f"{'M':>4} {'tiled':>7} {'amxbmm':>7} {'fp32avx':>7} | {'amx_spd':>7} {'avx_spd':>7} | "
          f"{'amx_exact':>9} {'amx_setk512':>11} {'avx_setk512':>11}")
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        l_tiled = il.indexer_logits(q, kv, w)
        l_amx = il.indexer_logits_amxbmm(q, kv, w)
        l_avx = il.indexer_logits_fp32avx(q, kv, w)
        t_tiled = tstream(lambda: il.indexer_logits(q, kv, w))
        t_amx = tstream(lambda: il.indexer_logits_amxbmm(q, kv, w))
        t_avx = tstream(lambda: il.indexer_logits_fp32avx(q, kv, w))
        amx_exact = bool(torch.equal(l_tiled, l_amx))
        amx_sm = _setmatch(l_tiled, l_amx, 512)
        avx_sm = _setmatch(l_tiled, l_avx, 512)
        print(f"{M:>4} {t_tiled:>7.1f} {t_amx:>7.1f} {t_avx:>7.1f} | {t_tiled / t_amx:>6.2f}x {t_tiled / t_avx:>6.2f}x | "
              f"{str(amx_exact):>9} {amx_sm:>11.3f} {avx_sm:>11.3f}")
    print("# amx_spd/avx_spd = shipped/variant (>1 = variant faster). amx_exact=True => single bf16 bmm is")
    print("# bit-identical to the shipped tiled path => conformance-SAFE structural win; ship adaptive by M.")


if __name__ == "__main__":
    main()
