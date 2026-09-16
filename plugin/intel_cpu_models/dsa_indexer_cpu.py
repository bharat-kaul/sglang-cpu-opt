"""CPU DSA lightning-indexer compute — Thesis-2 new-kernel-leg (reference-first).

The DeepSeek-V4 DSA indexer scores every KV token for a query and selects the top-k
to attend to (sparse attention). Two novel ops, no CPU kernel upstream:
  - `indexer_logits`: per-(query,kv) score = sum_heads( relu(q_h · kv) * weight_h )  ->  [N, S]
    (mirrors the in-tree torch fallback fp8_paged_mqa_logits_torch: relu, per-head weight, sum).
  - `indexer_topk`: select the top-`k` KV positions per query from the logits.
Each has a slow oracle + a vectorized CPU port validated against it.
"""
from __future__ import annotations

import torch


def indexer_logits_reference(q, kv, weight) -> torch.Tensor:
    # q: [N, H, D]; kv: [N, S, D]; weight: [N, H] -> logits [N, S]
    N, H, D = q.shape
    S = kv.shape[1]
    out = torch.zeros(N, S, dtype=torch.float32)
    for n in range(N):
        for s in range(S):
            acc = 0.0
            for h in range(H):
                acc += torch.relu((q[n, h].float() * kv[n, s].float()).sum()) * weight[n, h].float()
            out[n, s] = acc
    return out


def indexer_logits(q, kv, weight) -> torch.Tensor:
    """Vectorized CPU port. q:[N,H,D], kv:[N,S,D], weight:[N,H] -> [N,S]."""
    scores = torch.einsum("nhd,nsd->nhs", q.float(), kv.float())
    return (torch.relu(scores) * weight.float().unsqueeze(-1)).sum(dim=1)


def indexer_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k KV indices per query. logits:[N,S] -> indices [N, min(k,S)] (int64)."""
    k = min(k, logits.shape[1])
    return torch.topk(logits, k, dim=1, largest=True, sorted=False).indices
