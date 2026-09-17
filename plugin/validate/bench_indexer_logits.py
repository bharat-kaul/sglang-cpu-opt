"""Thesis-2 indexer-logits kernel: parity + speedup microbench.

Oracle = the current plugin path (fp32 einsum "nhd,sd->nhs" -> relu -> per-head
weight -> sum), i.e. _cpu_forward_c4_indexer step (4). Fast = the authored
formulation: fold the N*nh rows into one bf16 GEMM (q2d @ ck^T) so it dispatches
to oneDNN AMX BRGEMM (the M=16 index_gemm shape), then a fused relu * per-head
weight * sum epilogue. bf16 also matches the reference model, which computes the
indexer in bf16 (not fp32), so it is more faithful, not a regression.

Gate: top-k index SET must match the oracle (the logits only feed top-k
selection), and the fast path must be faster.
"""
import time

import torch

torch.manual_seed(0)


def oracle_fp32(q, ck, w):
    # q [N, nh, hd] float, ck [S, hd] float, w [N, nh] float
    scores = torch.einsum("nhd,sd->nhs", q, ck)  # [N, nh, S]
    return (torch.relu(scores) * w.unsqueeze(-1)).sum(dim=1)  # [N, S]


def fast_bf16(q, ck, w):
    # One GEMM over folded (N*nh) rows -> AMX BRGEMM; fused relu*weight*sum epilogue.
    N, nh, hd = q.shape
    S = ck.shape[0]
    qb = q.reshape(N * nh, hd).to(torch.bfloat16)
    ckb = ck.to(torch.bfloat16)
    scores = torch.mm(qb, ckb.t()).float().reshape(N, nh, S)  # AMX bf16 matmul
    return (torch.relu(scores) * w.unsqueeze(-1)).sum(dim=1)


def topk_set_match(a, b, k):
    ka = torch.topk(a, k, dim=1).indices
    kb = torch.topk(b, k, dim=1).indices
    inter = [len(set(ka[i].tolist()) & set(kb[i].tolist())) for i in range(a.shape[0])]
    return sum(inter) / (a.shape[0] * k)


def bench(fn, *args, iters=50):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


for N, nh, hd, S, topk in [(1, 64, 128, 512, 512), (1, 64, 128, 4096, 512), (8, 64, 128, 2048, 512)]:
    q = torch.randn(N, nh, hd)
    ck = torch.randn(S, hd)
    w = torch.rand(N, nh)
    k = min(topk, S)
    lo = oracle_fp32(q, ck, w)
    lf = fast_bf16(q, ck, w)
    match = topk_set_match(lo, lf, k)
    t_o = bench(oracle_fp32, q, ck, w)
    t_f = bench(fast_bf16, q, ck, w)
    print(
        f"N={N} nh={nh} hd={hd} S={S} topk={k}: topk-set-match={match*100:.1f}%  "
        f"oracle={t_o:.3f}ms  fast={t_f:.3f}ms  speedup={t_o/t_f:.2f}x"
    )
