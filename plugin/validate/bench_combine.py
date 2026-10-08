#!/usr/bin/env python3
"""Phase-A op #6 MHC combine/assign: parity (cosine vs torch oracle) + speedup.
Compiles plugin/kernels/dsa_pilot/combine.cpp and checks it against _cpu_hc_combine."""
import os
import time

import torch
from torch.utils.cpp_extension import load

torch.manual_seed(0)
K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "combine.cpp")
mod = load(name="combine_pilot", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)

HC, H = 4, 4096  # hc clusters, hidden


def ref_combine(x_flat, pre, hc):
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    xr = x_flat.reshape(m, hc, h).float()
    return torch.einsum("mk,mkh->mh", pre.float(), xr)


def bench(fn, *a, it=50):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


print(f"op#6 MHC combine  hc={HC} H={H}")
print(f"{'M':>4} {'cos':>10} {'ref_ms':>9} {'cpp_ms':>9} {'speedup':>8}")
for M in (1, 8, 16, 32, 64):
    x_flat = torch.randn(M, HC * H)
    pre = torch.rand(M, HC)
    r = ref_combine(x_flat, pre, HC)
    g = mod.mhc_combine(x_flat, pre, HC)
    cos = torch.nn.functional.cosine_similarity(r.flatten(), g.flatten(), dim=0).item()
    t_ref = bench(ref_combine, x_flat, pre, HC)
    t_cpp = bench(mod.mhc_combine, x_flat, pre, HC)
    print(f"{M:>4} {cos:>10.6f} {t_ref:>9.3f} {t_cpp:>9.3f} {t_ref/t_cpp:>7.2f}x")
