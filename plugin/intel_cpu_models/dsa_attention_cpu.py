"""CPU DSA attention — the three validated DSA kernels composed into one forward.

Thesis-2 capstone: the full DeepSeek-V4 sparse-attention *compute* on CPU, end to end:
  index (score every KV token) -> select top-k -> sparse-attend over the selected tokens.
Correctness anchor: with top-k >= seq_len the selection is "all tokens", so DSA attention
MUST reduce EXACTLY to dense attention — a strong end-to-end check that the composition
(not just each kernel) is right. The compressor kernel (dsa_compressor_cpu) is the upstream
KV-compression the indexer scores over and is validated separately.

This proves the novel DSA math runs correctly on CPU. It does NOT include the paged
flash-MLA serving machinery (compress plan byte-layout + state-pool ring + paged fp8
gather/dequant) — that runtime integration is the remaining engineering to a full serve.
"""
from __future__ import annotations

import math

import torch

from intel_cpu_models.dsa_indexer_cpu import indexer_logits, indexer_topk


def dsa_attention(q, k, v, index_weight, topk, scale=None):
    """q:[N,H,D]; k:[N,S,D]; v:[N,S,Dv]; index_weight:[N,H] -> [N,H,Dv].

    index (indexer_logits) -> top-k select -> sparse scaled-dot-product attention.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    logits = indexer_logits(q, k, index_weight)  # [N, S]
    idx = indexer_topk(logits, topk)  # [N, k]
    k_sel = torch.gather(k, 1, idx.unsqueeze(-1).expand(-1, -1, k.shape[-1]))
    v_sel = torch.gather(v, 1, idx.unsqueeze(-1).expand(-1, -1, v.shape[-1]))
    s = torch.einsum("nhd,nkd->nhk", q.float(), k_sel.float()) * scale
    w = s.softmax(dim=-1)
    return torch.einsum("nhk,nkv->nhv", w, v_sel.float())


def dense_attention_reference(q, k, v, scale=None):
    """Dense scaled-dot-product attention over ALL tokens (the anchor)."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = torch.einsum("nhd,nkd->nhk", q.float(), k.float()) * scale
    w = s.softmax(dim=-1)
    return torch.einsum("nhk,nkv->nhv", w, v.float())
