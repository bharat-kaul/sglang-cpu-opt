#!/usr/bin/env python3
"""Phase-A op #3 DSA compressor (softmax-pool): parity (cosine vs torch oracle) + speedup.
Compiles plugin/kernels/dsa_pilot/compressor.cpp and checks it against the reference."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "intel_cpu_models"))
from dsa_compressor_cpu import compress_softmax_pool  # torch reference
from torch.utils.cpp_extension import load

torch.manual_seed(0)
K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "compressor.cpp")
mod = load(name="compressor_pilot", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)


def bench(fn, *a, it=50):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


R, D = 128, 512  # c128 window (ratio-128 non-overlap), head_dim=512 (main attn compressor; published model.py)
print(f"op#3 compressor softmax-pool  R={R} D={D}")
print(f"{'M':>4} {'cos':>10} {'ref_ms':>9} {'cpp_ms':>9} {'speedup':>8}")
for M in (1, 8, 16, 32, 64):
    kv = torch.randn(M, R, D)
    score = torch.randn(M, R, D)
    ape = torch.randn(R, D)
    ref = compress_softmax_pool(kv, score, ape)
    got = mod.compressor_softmax_pool(kv, score, ape)
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    t_ref = bench(compress_softmax_pool, kv, score, ape)
    t_cpp = bench(mod.compressor_softmax_pool, kv, score, ape)
    print(f"{M:>4} {cos:>10.6f} {t_ref:>9.3f} {t_cpp:>9.3f} {t_ref/t_cpp:>7.2f}x")
