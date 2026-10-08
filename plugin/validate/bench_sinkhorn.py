#!/usr/bin/env python3
"""Phase-A op #5 MHC sinkhorn: parity (cosine vs torch oracle) + speedup.
Compiles plugin/kernels/dsa_pilot/sinkhorn.cpp and checks it against the sglang reference."""
import os
import time

import torch
from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch  # torch reference
from torch.utils.cpp_extension import load

torch.manual_seed(0)
K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "sinkhorn.cpp")
mod = load(name="sinkhorn_pilot", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)

HC, ITERS, EPS = 4, 20, 1e-6


def bench(fn, *a, it=50):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


print(f"op#5 MHC sinkhorn  hc={HC} iters={ITERS}")
print(f"{'M':>4} {'cos_pre':>9} {'cos_post':>9} {'cos_comb':>9} {'ref_ms':>9} {'cpp_ms':>9} {'speedup':>8}")
for M in (1, 8, 16, 32, 64):
    W = (2 + HC) * HC
    mixes = torch.randn(1, M, W)
    scale = torch.rand(3) + 0.5
    base = torch.randn(W)
    rp, ro, rc = _hc_split_sinkhorn_torch(mixes, scale, base, HC, ITERS, EPS)
    gp, go, gc = mod.mhc_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
    cp = torch.nn.functional.cosine_similarity(rp.flatten().float(), gp.flatten().float(), dim=0).item()
    co = torch.nn.functional.cosine_similarity(ro.flatten().float(), go.flatten().float(), dim=0).item()
    cc = torch.nn.functional.cosine_similarity(rc.flatten().float(), gc.flatten().float(), dim=0).item()
    t_ref = bench(_hc_split_sinkhorn_torch, mixes, scale, base, HC, ITERS, EPS)
    t_cpp = bench(mod.mhc_sinkhorn, mixes, scale, base, HC, ITERS, EPS)
    print(f"{M:>4} {cp:>9.6f} {co:>9.6f} {cc:>9.6f} {t_ref:>9.3f} {t_cpp:>9.3f} {t_ref/t_cpp:>7.2f}x")
