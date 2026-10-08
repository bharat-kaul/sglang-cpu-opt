#!/usr/bin/env python3
"""Phase-A op #4 DSA sparse attend (top-k KV): parity (cosine vs torch oracle) + speedup.
Compiles plugin/kernels/dsa_pilot/sparse_attend.cpp and checks it against the reference."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "intel_cpu_models"))
from dsa_sparse_attention_cpu import sparse_attention  # torch reference
from torch.utils.cpp_extension import load

torch.manual_seed(0)
K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "sparse_attend.cpp")
mod = load(name="sparse_attend_pilot", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)


def bench(fn, *a, it=30):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


H, Ktop, D, Dv = 8, 512, 128, 128  # heads, top-k KV, head dims
print(f"op#4 sparse attend  H={H} Ktop={Ktop} D={D} Dv={Dv}")
print(f"{'M':>4} {'cos_sc':>9} {'cos_amx':>9} {'ref_ms':>8} {'sc_ms':>8} {'amx_ms':>8} {'sc_x':>6} {'amx_x':>6}")
for M in (1, 8, 16, 32, 64):
    q = torch.randn(M, H, D)
    k = torch.randn(M, H, Ktop, D)
    v = torch.randn(M, H, Ktop, Dv)
    ref = sparse_attention(q, k, v)
    sc = mod.sparse_attend(q, k, v, 0.0)
    amx = mod.sparse_attend_amx(q, k, v, 0.0)
    cos_sc = torch.nn.functional.cosine_similarity(ref.flatten(), sc.flatten(), dim=0).item()
    cos_amx = torch.nn.functional.cosine_similarity(ref.flatten(), amx.flatten(), dim=0).item()
    t_ref = bench(sparse_attention, q, k, v)
    t_sc = bench(mod.sparse_attend, q, k, v, 0.0)
    t_amx = bench(mod.sparse_attend_amx, q, k, v, 0.0)
    print(f"{M:>4} {cos_sc:>9.6f} {cos_amx:>9.6f} {t_ref:>8.3f} {t_sc:>8.3f} {t_amx:>8.3f} "
          f"{t_ref/t_sc:>5.2f}x {t_ref/t_amx:>5.2f}x")
