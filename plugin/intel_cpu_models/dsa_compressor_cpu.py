"""CPU DSA compressor compute (softmax-pool) — Thesis-2 new-kernel-leg exemplar.

The DeepSeek-V4 DSA "lightning" compressor reduces a window of `ratio` raw KV tokens
(ratio = 4 or 128) to ONE compressed token via a per-channel softmax-weighted pool:
weights = softmax over the window of (score + absolute-position-embedding `ape`);
out = sum_window(weights * kv). No CPU kernel exists upstream (CUDA/TileLang only).

Authored reference-first per `kernel-authoring` §1b: `..._reference` is the slow,
obviously-correct numerical ORACLE; `compress_softmax_pool` is the vectorized CPU port
validated against it (and it becomes the oracle for any future AMX version). Math mirrors
the in-tree HIP torch fallback `_compress_forward_c128_fallback` (compressor_v2.py).
"""
from __future__ import annotations

import torch


def compress_softmax_pool_reference(
    kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """Slow oracle. kv, score: [N, ratio, D]; ape: [ratio, D] -> out [N, D]."""
    N, R, D = kv.shape
    out = torch.zeros(N, D, dtype=torch.float32)
    for n in range(N):
        for d in range(D):
            w = torch.softmax(score[n, :, d].float() + ape[:, d].float(), dim=0)
            out[n, d] = (w * kv[n, :, d].float()).sum()
    return out


def compress_softmax_pool(
    kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """Vectorized CPU port (same math, no python loop). [N, ratio, D] -> [N, D]."""
    w = (score.float() + ape.float().unsqueeze(0)).softmax(dim=1)
    return (w * kv.float()).sum(dim=1)
