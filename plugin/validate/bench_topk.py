#!/usr/bin/env python3
"""Phase-A op #2 DSA indexer top-k: set-match (vs torch.topk) + speedup.
Compiles plugin/kernels/dsa_pilot/indexer_topk.cpp. Tests the always-C/C++ best-of
(row-parallel large N, chunked 2-pass small N) vs torch.topk at every M."""
import os
import time

import torch
from torch.utils.cpp_extension import load

torch.manual_seed(0)
K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "indexer_topk.cpp")
mod = load(name="indexer_topk_pilot", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)

S, k = 1024, 512  # compressed index context (~seqlen/ratio4), verdict #6


def setmatch(a, b):
    return sum(len(set(a[i].tolist()) & set(b[i].tolist())) for i in range(a.shape[0])) / (a.shape[0] * b.shape[1])


def bench(fn, *a, it=50):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


print(f"op#2 indexer top-k  S={S} k={k}")
print(f"{'M':>4} {'set_match':>10} {'torch_ms':>9} {'cpp_ms':>9} {'speedup':>8}")
for M in (1, 8, 16, 32, 64):
    logits = torch.randn(M, S)
    ref = torch.topk(logits, k, dim=1, sorted=False).indices
    got = mod.indexer_topk(logits, k)
    sm = setmatch(got, ref)
    t_torch = bench(lambda x: torch.topk(x, k, dim=1, sorted=False).indices, logits)
    t_cpp = bench(mod.indexer_topk, logits, k)
    print(f"{M:>4} {sm:>10.4f} {t_torch:>9.3f} {t_cpp:>9.3f} {t_torch/t_cpp:>7.2f}x")
