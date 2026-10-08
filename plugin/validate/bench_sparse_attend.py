#!/usr/bin/env python3
"""Phase-A op #4 DSA MLA sparse attend: parity (cosine vs published-math oracle) + speedup.
Shapes GROUNDED in DeepSeek-V4-Flash model.py: MQA (1 kv latent), H=num_attention_heads=64,
D=head_dim=512 (k==v), per-head attn_sink. Compiles sparse_attend.cpp and checks scalar + AMX."""
import os
import time

import torch
from torch.utils.cpp_extension import load

torch.manual_seed(0)
K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "sparse_attend.cpp")
mod = load(name="sparse_attend_pilot", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)

H, Ktop, D = 64, 512, 512  # num_attention_heads, index_topk, head_dim (k==v latent)
SCALE = D ** -0.5


def ref_sparse_attn(q, kv, sink, scale):
    # published math: scores=q@kv^T*scale; softmax with per-head sink; out=w@kv
    scores = torch.einsum("nhd,nkd->nhk", q.float(), kv.float()) * scale
    m = torch.maximum(scores.max(-1, keepdim=True).values, sink.view(1, H, 1))
    e = (scores - m).exp()
    denom = e.sum(-1, keepdim=True) + (sink.view(1, H, 1) - m).exp()
    w = e / denom
    return torch.einsum("nhk,nkd->nhd", w, kv.float())


def bench(fn, *a, it=20):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


print(f"op#4 MLA sparse attend (MQA+sink)  H={H} Ktop={Ktop} D={D}")
print(f"{'M':>4} {'cos_sc':>9} {'cos_amx':>9} {'ref_ms':>8} {'sc_ms':>8} {'amx_ms':>8} {'sc_x':>6} {'amx_x':>6}")
for M in (1, 8, 16, 32, 64):
    q = torch.randn(M, H, D)
    kv = torch.randn(M, Ktop, D)          # MQA: one latent kv shared across H heads
    sink = torch.randn(H)
    ref = ref_sparse_attn(q, kv, sink, SCALE)
    sc = mod.sparse_attend(q, kv, sink, SCALE)
    amx = mod.sparse_attend_amx(q, kv, sink, SCALE)
    cos_sc = torch.nn.functional.cosine_similarity(ref.flatten(), sc.flatten(), dim=0).item()
    cos_amx = torch.nn.functional.cosine_similarity(ref.flatten(), amx.flatten(), dim=0).item()
    t_ref = bench(ref_sparse_attn, q, kv, sink, SCALE)
    t_sc = bench(mod.sparse_attend, q, kv, sink, SCALE)
    t_amx = bench(mod.sparse_attend_amx, q, kv, sink, SCALE)
    print(f"{M:>4} {cos_sc:>9.6f} {cos_amx:>9.6f} {t_ref:>8.3f} {t_sc:>8.3f} {t_amx:>8.3f} "
          f"{t_ref/t_sc:>5.2f}x {t_ref/t_amx:>5.2f}x")
