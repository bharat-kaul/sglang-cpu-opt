"""Source-faithful torch replicas of the DeepSeek-V4-Flash sparse_attn primitive
(inference/kernel.py sparse_attn_kernel @ pin 60d8d70). SINGLE source of truth shared by the F4 gate and
the GPU oracle so the two never drift. Shapes: q[N,H,D], kv[N,K,D] (MQA shared latent, k==v), sink[H] fp32,
scale -> out[N,H,D].
"""
import torch


def ora_sparse_fp32(q, kv, sink, scale):
    """FP32 diagnostic: global softmax, fp32 throughout (NOT the serving dtype)."""
    hh = q.shape[1]
    scores = torch.einsum("nhd,nkd->nhk", q.float(), kv.float()) * scale
    m = torch.maximum(scores.max(-1, keepdim=True).values, sink.view(1, hh, 1))
    e = (scores - m).exp()
    denom = e.sum(-1, keepdim=True) + (sink.view(1, hh, 1) - m).exp()
    return torch.einsum("nhk,nkd->nhd", e / denom, kv.float())


def ora_sparse_bf16_global(q, kv, sink, scale):
    """BF16 GLOBAL replica (NOT source-faithful: one global softmax + one bf16 cast of all weights, sink in
    the max). Retained only as a diagnostic; prefer ora_sparse_blockwise."""
    hh = q.shape[1]
    qb, kvb = q.bfloat16().float(), kv.bfloat16().float()
    scores = torch.einsum("nhd,nkd->nhk", qb, kvb) * scale
    m = torch.maximum(scores.max(-1, keepdim=True).values, sink.view(1, hh, 1))
    e = (scores - m).exp()
    denom = e.sum(-1, keepdim=True) + (sink.view(1, hh, 1) - m).exp()
    w = e.bfloat16().float()
    return (torch.einsum("nhk,nkd->nhd", w, kvb) / denom).bfloat16().float()


def ora_sparse_blockwise(q, kv, sink, scale, block=64):
    """SOURCE-FAITHFUL replica of sparse_attn_kernel: FlashAttention-style online softmax over K in blocks of
    64, bf16 operands with FP32 accumulate in BOTH GEMMs, a BF16 cast of the exp'd weights before the value
    GEMM, per-block FP32 rescale of the running sum/output, the sink added AFTER the loop using the FINAL
    running max, and a BF16 output. This matches the kernel's op/rounding ORDER, which a single global
    softmax does not (the published kernel bf16-rounds each block's exponentials separately)."""
    N, H, D = q.shape
    K = kv.shape[1]
    dev = q.device                     # run the running-state on the inputs' device (CPU or CUDA)
    qb = q.bfloat16().float()          # bf16-rounded operands; the GEMM accumulates these in fp32
    kvb = kv.bfloat16().float()
    sk = sink.float()
    out = torch.empty(N, H, D, dtype=torch.float32, device=dev)
    for n in range(N):
        m = torch.full((H,), float("-inf"), device=dev)
        l = torch.zeros(H, device=dev)                             # running sum_exp
        acc = torch.zeros(H, D, device=dev)                        # running acc_o (fp32)
        qn = qb[n]
        for s0 in range(0, K, block):
            s1 = min(s0 + block, K)
            kvblk = kvb[n, s0:s1]                                   # [blk, D]
            sblk = (qn @ kvblk.transpose(0, 1)) * scale            # [H, blk] bf16-operand, fp32 accumulate
            m_prev = m
            m = torch.maximum(m_prev, sblk.max(dim=1).values)
            resc = torch.exp(m_prev - m)                            # 0 on the first block (m_prev = -inf)
            resc = torch.where(torch.isfinite(resc), resc, torch.zeros_like(resc))
            e = torch.exp(sblk - m.unsqueeze(1))                    # [H, blk]
            l = l * resc + e.sum(dim=1)
            wblk = e.bfloat16().float()                            # BF16 cast of exp'd weights before value GEMM
            acc = acc * resc.unsqueeze(1) + (wblk @ kvblk)         # bf16-operand @ bf16-operand, fp32 accumulate
        l = l + torch.exp(sk - m)                                  # sink post-loop with the FINAL running max
        out[n] = acc / l.unsqueeze(1)
    return out.bfloat16().float()                                  # bf16 output boundary
