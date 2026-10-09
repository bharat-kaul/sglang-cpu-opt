"""Sparse-attention GPU-oracle parity: the REAL TileLang kernel (DeepSeek-V4-Flash
inference/kernel.py sparse_attn) vs our two torch oracle replicas — the FP32 math
(_ora_sparse, diagnostic) and the BF16 primitive replica (_ora_sparse_bf16, the F4
authoritative). Resolves the OPEN conformance question: is the published sparse_attn
bf16 (=> the fp32 F4 oracle is over-strict and bf16-AMX is faithful) or fp32?

Mirrors test_kda_gpu_oracle.py: the real kernel is the TRUTH; the torch replica is the
candidate. Runs on an H200 in the lmsysorg/sglang container (needs tilelang + CUDA).

Also SAVES the exact seeded inputs + the GPU output to /scratch so the CPU side (native
AMX) can diff our compiled sparse_attend scalar/fp32bmm/amx against the real kernel.

MQA equivalence to the CPU kernel sparse_attend(q[N,H,D], kv[N,K,D], sink[H], scale):
  GPU q[b,m,h,d], kv[b,n,d] shared, topk_idxs[b,m,topk]. Set b=1, m=N, n=topk=K and
  topk_idxs = arange(K) for every m => every query attends the whole K-row pool. The CPU
  kv[N,K,D] is that same pool broadcast across N (all queries share the gathered set).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/scratch/bkaul/models/DeepSeek-V4-Flash")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference.kernel import sparse_attn  # noqa: E402  (real TileLang kernel)
from sparse_ref import (ora_sparse_fp32 as _ora_sparse,          # noqa: E402  (single source of truth)
                        ora_sparse_bf16_global as _ora_sparse_bf16,
                        ora_sparse_blockwise as _ora_sparse_blockwise)

H, D, K = 64, 512, 512          # real dims: heads, head_dim, top-k (== kv pool size here)
SAVE = "/scratch/bkaul/sparse_oracle_io.pt"


def metrics(ref, got):
    r, g = ref.flatten().float(), got.flatten().float()
    cos = F.cosine_similarity(r, g, dim=0).item()
    mae = (r - g).abs().max().item()
    ratio = (g.norm() / r.norm()).item()
    return cos, mae, ratio


def run(N, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    scale = D ** -0.5
    # GPU inputs: b=1, m=N, shared kv pool of K rows, every query attends the full pool.
    q = torch.randn(1, N, H, D, device=dev, dtype=torch.float32)
    kv_pool = torch.randn(1, K, D, device=dev, dtype=torch.float32)       # [1,K,D] shared
    sink = torch.randn(H, device=dev, dtype=torch.float32)
    idxs = torch.arange(K, device=dev, dtype=torch.int32).view(1, 1, K).expand(1, N, K).contiguous()

    o_gpu = sparse_attn(q.bfloat16(), kv_pool.bfloat16(), sink, idxs, scale)  # [1,N,H,D] bf16
    o_gpu = o_gpu.reshape(N, H, D).float()

    # CPU-equivalent layout: q[N,H,D], kv[N,K,D] = pool broadcast across N.
    qc = q.reshape(N, H, D)
    kvc = kv_pool.reshape(K, D).unsqueeze(0).expand(N, K, D).contiguous()
    c32 = metrics(o_gpu, _ora_sparse(qc, kvc, sink, scale))
    c16 = metrics(o_gpu, _ora_sparse_bf16(qc, kvc, sink, scale))
    cblk = metrics(o_gpu, _ora_sparse_blockwise(qc, kvc, sink, scale))     # SOURCE-FAITHFUL 64-block replica
    print(f"N={N:>3}  fp32 vs GPU: mae={c32[1]:.3e}   bf16-global vs GPU: mae={c16[1]:.3e}   "
          f"blockwise vs GPU: cos={cblk[0]:.8f} mae={cblk[1]:.3e} ratio={cblk[2]:.5f}")
    return {"N": N, "scale": scale, "q": qc.cpu(), "kv": kvc.cpu(), "sink": sink.cpu(),
            "gpu_out": o_gpu.cpu(), "fp32_vs_gpu": c32, "bf16_vs_gpu": c16, "blockwise_vs_gpu": cblk}


def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    recs = [run(N, seed=s) for N, s in ((1, 0), (8, 1), (64, 2))]
    # Provenance stamp (P1-F7/R2-F4): bind the saved io to the exact kernel source + run identity, and
    # VALIDATE the source hash fail-closed against the expected pin before trusting the run.
    import hashlib
    import socket
    ksrc = "/scratch/bkaul/models/DeepSeek-V4-Flash/inference/kernel.py"
    kern_sha = hashlib.sha256(open(ksrc, "rb").read()).hexdigest()
    expect = os.environ.get("ORACLE_KERNEL_SHA")
    if expect and kern_sha != expect:
        raise SystemExit(f"FATAL: kernel.py sha {kern_sha} != expected pin {expect}")
    prov = {"slurm_job_id": os.environ.get("SLURM_JOB_ID") or "interactive", "host": socket.gethostname(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "model_snapshot": ksrc,
            "kernel_py_sha256": kern_sha, "kernel_py_sha256_validated": bool(expect) and kern_sha == expect,
            "container_image": os.environ.get("ORACLE_IMG") or "lmsysorg/sglang:latest(unverified)"}
    torch.save({"provenance": prov, "records": recs}, SAVE)
    print(f"[provenance] {prov}")
    print(f"[saved io -> {SAVE}]")
    # Verdict: the replica with the TIGHTER max-abs-err vs the real kernel is the authoritative dtype.
    worst16 = max(r["bf16_vs_gpu"][1] for r in recs)
    worst32 = max(r["fp32_vs_gpu"][1] for r in recs)
    verdict = "BF16 (bf16-replica is the closer reference -> fp32 oracle is over-strict; bf16-AMX is faithful)" \
        if worst16 <= worst32 else "FP32 (fp32 oracle is the closer reference)"
    print(f"[SPARSE ORACLE] worst mae: bf16-replica={worst16:.3e}  fp32-oracle={worst32:.3e} -> reference dtype = {verdict}")
    coverage()


def coverage():
    """R2-F5 coverage: INDEPENDENT batches (b=N, m=1 -> own kv per request), variable topk K in {128,160,640},
    and causal SENTINELS (-1 padding entries the kernel masks). Compare the real kernel to the source-faithful
    blockwise replica on the GATHERED kv (the rows each request actually selects)."""
    dev = "cuda"
    print(">>> SPARSE COVERAGE (independent batches + K-unions + sentinels) vs blockwise replica")
    for Ktop, sent, tag in ((128, 0, "K128"), (160, 0, "K160"), (640, 0, "K640"), (512, 64, "K512+64sentinels")):
        torch.manual_seed(1000 + Ktop + sent)
        B, Kpool = 4, max(1024, Ktop + 64)
        scale = D ** -0.5
        q = torch.randn(B, 1, H, D, device=dev, dtype=torch.float32)
        kv = torch.randn(B, Kpool, D, device=dev, dtype=torch.float32)        # INDEPENDENT pool per request
        sink = torch.randn(H, device=dev, dtype=torch.float32)
        idxs = torch.empty(B, 1, Ktop, device=dev, dtype=torch.int32)
        for b in range(B):
            perm = torch.randperm(Kpool, device=dev)[:Ktop].to(torch.int32)
            if sent:
                perm[-sent:] = -1                                            # causal/padding sentinels (masked)
            idxs[b, 0] = perm
        o_gpu = sparse_attn(q.bfloat16(), kv.bfloat16(), sink, idxs, scale).reshape(B, H, D).float()
        # CPU-equivalent gather: each request attends ONLY its valid (non -1) selected rows.
        maes = []
        for b in range(B):
            val = idxs[b, 0][idxs[b, 0] >= 0].long()
            kvg = kv[b, val].unsqueeze(0)                                     # [1, |val|, D]
            ref = _ora_sparse_blockwise(q[b, 0].unsqueeze(0), kvg, sink, scale)  # [1,H,D]
            maes.append((o_gpu[b].unsqueeze(0) - ref).abs().max().item())
        print(f"    {tag:>18}  B={B} Kpool={Kpool} topk={Ktop} sentinels={sent}  "
              f"blockwise mae max={max(maes):.3e} mean={sum(maes)/len(maes):.3e}")


if __name__ == "__main__":
    main()
