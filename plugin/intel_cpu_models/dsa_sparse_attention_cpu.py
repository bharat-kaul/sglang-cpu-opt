"""CPU DSA sparse-prefill attention compute — Thesis-2 new-kernel-leg (reference-first).

After the indexer selects the top-k KV positions, DSA attends only to those (sparse).
This is scaled-dot-product attention over the gathered tokens; no CPU kernel upstream
(the sparse-prefill path is Triton-only). Slow oracle + vectorized CPU port, validated.
"""
from __future__ import annotations

import math

import torch


def sparse_attention_reference(q, k, v, scale=None) -> torch.Tensor:
    # q:[N,H,D]; k:[N,H,K,D]; v:[N,H,K,Dv] -> [N,H,Dv]
    N, H, D = q.shape
    Dv = v.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    out = torch.zeros(N, H, Dv, dtype=torch.float32)
    for n in range(N):
        for h in range(H):
            s = (q[n, h].float().unsqueeze(0) * k[n, h].float()).sum(-1) * scale
            w = torch.softmax(s, dim=0)
            out[n, h] = (w.unsqueeze(-1) * v[n, h].float()).sum(0)
    return out


def sparse_attention(q, k, v, scale=None) -> torch.Tensor:
    """Vectorized CPU port. q:[N,H,D], k:[N,H,K,D], v:[N,H,K,Dv] -> [N,H,Dv]."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("nhd,nhkd->nhk", q.float(), k.float()) * scale
    w = scores.softmax(dim=-1)
    return torch.einsum("nhk,nhkv->nhv", w, v.float())
