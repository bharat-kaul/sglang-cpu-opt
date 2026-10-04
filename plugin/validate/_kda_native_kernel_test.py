"""DECISIVE: does the native AMX CPU kernel chunk_gated_delta_rule_cpu match the KDA
reference (kda_recurrent)? If yes, GLM KDA can WIRE the fused kernel instead of the torch
chunked impl -> roofline-level win using existing sgl-kernel code. Login/EMR, no weights.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "intel_cpu_models"))
from kda_linear_attention_cpu import kda_recurrent, kda_sigmoid_gate  # noqa: E402

from sgl_kernel.mamba import chunk_gated_delta_rule_cpu  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-30)).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), 0).item()


def run(T, H, K, V, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, H, K); k = torch.randn(T, H, K); v = torch.randn(T, H, V)
    a = torch.randn(T, H, K); dt = torch.randn(H, K) * 0.1; Al = torch.randn(H, 1) * 0.1
    g = kda_sigmoid_gate(a, dt.unsqueeze(0), Al.unsqueeze(0), -5.0)   # [T,H,K] per-key log-decay
    beta = torch.sigmoid(torch.randn(T, H))
    # reference (applies l2norm(q,k) + scale=K**-0.5 internally)
    o_ref, s_ref = kda_recurrent(q, k, v, g, beta)
    # kernel: head_first=False -> [B,T,H,D]; single varlen seq cu_seqlens=[0,T]
    qb = q.unsqueeze(0).to(dtype); kb = k.unsqueeze(0).to(dtype); vb = v.unsqueeze(0).to(dtype)
    gb = g.unsqueeze(0).float(); bb = beta.unsqueeze(0).float()
    init = torch.zeros(1, H, K, V, dtype=torch.float32)
    cu = torch.tensor([0, T], dtype=torch.int32)
    idx = torch.tensor([0], dtype=torch.int32)
    for (osf, label) in [(True, "")]:
        try:
            o_k, s_k = chunk_gated_delta_rule_cpu(qb, kb, vb, gb, bb, init, cu, False, True, idx)
            ok = o_k.reshape(T, H, V)
            print(f"  T={T} H={H} K={K} {dtype}: out cos={cos(ok,o_ref):.5f} rel={rel(ok,o_ref):.2e} "
                  f"| state cos={cos(s_k,s_ref):.5f} rel={rel(s_k,s_ref):.2e} | kshape={tuple(o_k.shape)}")
        except Exception as e:
            print(f"  T={T} H={H} K={K} {dtype}: ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    print("chunk_gated_delta_rule_cpu vs kda_recurrent (KDA oracle):")
    for dt in (torch.bfloat16, torch.float32):
        for (T, H, K, V) in [(64, 8, 128, 128), (256, 16, 128, 128)]:
            run(T, H, K, V, dt)
