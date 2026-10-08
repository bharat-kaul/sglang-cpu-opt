#!/usr/bin/env python3
"""Phase-A op #2: DSA indexer top-k. Race custom C++ nth_element vs torch.topk; verify the
selected SET matches (unsorted top-k is all sparse attention needs)."""
import os
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "64")))

mod = load(name="dsa_topk_pilot",
           sources=[os.path.join(ROOT, "kernels", "dsa_pilot", "indexer_topk.cpp")],
           extra_cflags=["-O3", "-march=native", "-fopenmp", "-funsafe-math-optimizations"],
           verbose=False)

S, K = 4096, 512


def bench(fn, *a, it=30):
    for _ in range(5):
        fn(*a)
    ts = []
    for _ in range(it):
        t = time.perf_counter(); fn(*a); ts.append(time.perf_counter() - t)
    return min(ts)


print(f"{'N':>4} {'setmatch':>9} {'torch ms':>9} {'cpp ms':>9} {'speedup':>8}")
for N in (1, 8, 16, 32, 64):
    lg = torch.randn(N, S)
    ref = torch.topk(lg, K, dim=1, largest=True, sorted=False).indices
    cpp = mod.indexer_topk(lg, K)
    # set-equality per row (order-independent)
    match = all(set(ref[n].tolist()) == set(cpp[n].tolist()) for n in range(N))
    t_t = bench(lambda x: torch.topk(x, K, dim=1, largest=True, sorted=False).indices, lg)
    t_c = bench(mod.indexer_topk, lg, K)
    print(f"{N:>4} {str(match):>9} {t_t*1e3:>9.3f} {t_c*1e3:>9.3f} {t_t/t_c:>7.2f}x")
