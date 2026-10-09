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


def ref_fp32_nonconformed(q, kv, weight):
    # LEGACY fp32-stage reference (DIFFERENT contract: fp32 reduce). Kept only as a secondary diagnostic
    # column; it is NOT the no-regression comparator (R2-F3: the comparator must share the kernel's contract).
    scores = torch.einsum("nhd,nsd->nhs", q.float(), kv.float())
    return (torch.relu(scores) * weight.unsqueeze(-1).float()).sum(1)


def ref_conformed(q, kv, weight):
    # SAME-CONTRACT fallback = the published bf16 stage boundaries (model.py L420-421): bf16 einsum output,
    # bf16 relu, bf16 SIGNED weights, bf16 reduce over heads -> bf16 logits. This is the arithmetic the C++
    # kernel implements, so its speedup is the apples-to-apples no-regression number (R2-F3).
    scores = torch.einsum("nhd,nsd->nhs", q.bfloat16(), kv.bfloat16())
    return (scores.relu_() * weight.bfloat16().unsqueeze(-1)).sum(1)


def setmatch(a, b, k=512):
    ia = torch.topk(a, k, dim=1, sorted=False).indices
    ib = torch.topk(b, k, dim=1, sorted=False).indices
    return sum(len(set(ia[i].tolist()) & set(ib[i].tolist())) for i in range(a.shape[0])) / (a.shape[0] * k)


def bench(fn, *a, it=30):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


print(f"op#1 indexer logits  H={H} D={D} S(compressed)={S}  (SIGNED weights; same-contract bf16 fallback)")
print(f"{'M':>4} {'setmatch':>9} {'conf_ms':>9} {'cpp_ms':>9} {'spd_conf':>9} {'fp32ref_ms':>11}")
for M in (1, 8, 16, 32, 64):
    q = torch.randn(M, H, D)
    kv = torch.randn(M, S, D)
    w = torch.randn(M, H)                                   # SIGNED (published weights_proj is a bf16 Linear)
    rc = ref_conformed(q, kv, w)
    g = mod.indexer_logits(q, kv, w)
    sm = setmatch(rc, g)                                   # same-contract top-512 set-match (correctness)
    t_conf = bench(ref_conformed, q, kv, w)                # SAME-CONTRACT no-regression comparator
    t_cpp = bench(mod.indexer_logits, q, kv, w)
    t_fp32 = bench(ref_fp32_nonconformed, q, kv, w)        # secondary (different contract)
    print(f"{M:>4} {sm:>9.4f} {t_conf:>9.3f} {t_cpp:>9.3f} {t_conf/t_cpp:>8.2f}x {t_fp32:>11.3f}")
