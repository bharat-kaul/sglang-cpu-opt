#!/usr/bin/env python3
"""Phase-A op #1 DSA indexer logits: parity (cosine vs reference) + speedup.
Context GROUNDED (verdict #6): indexer scores over the COMPRESSED index context
(~seqlen/ratio4 = 1024), NOT raw 4096. Compiles indexer_logits.cpp (best-of)."""
import os
import time

import torch
from torch.utils.cpp_extension import load

torch.manual_seed(0)
K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "indexer_logits.cpp")
mod = load(name="indexer_logits_pilot", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)

H, D, S = 64, 128, 1024  # index_n_heads, index_head_dim, COMPRESSED index context (~seqlen/4)


def ref(q, kv, weight):
    # logits[n,s] = sum_h relu(q[n,h,:].kv[n,s,:]) * weight[n,h]
    scores = torch.einsum("nhd,nsd->nhs", q.float(), kv.float())
    return (torch.relu(scores) * weight.unsqueeze(-1).float()).sum(1)


def bench(fn, *a, it=30):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


print(f"op#1 indexer logits  H={H} D={D} S(compressed)={S}")
print(f"{'M':>4} {'cos':>10} {'ref_ms':>9} {'cpp_ms':>9} {'speedup':>8}")
for M in (1, 8, 16, 32, 64):
    q = torch.randn(M, H, D)
    kv = torch.randn(M, S, D)
    w = torch.rand(M, H)
    r = ref(q, kv, w)
    g = mod.indexer_logits(q, kv, w)
    cos = torch.nn.functional.cosine_similarity(r.flatten(), g.flatten(), dim=0).item()
    t_ref = bench(ref, q, kv, w)
    t_cpp = bench(mod.indexer_logits, q, kv, w)
    print(f"{M:>4} {cos:>10.6f} {t_ref:>9.3f} {t_cpp:>9.3f} {t_ref/t_cpp:>7.2f}x")
