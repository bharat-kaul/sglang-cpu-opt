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
from inference.kernel import sparse_attn  # noqa: E402  (real TileLang kernel)

H, D, K = 64, 512, 512          # real dims: heads, head_dim, top-k (== kv pool size here)
SAVE = "/scratch/bkaul/sparse_oracle_io.pt"


def _ora_sparse(q, kv, sink, scale):                       # FP32 math (diagnostic)
    hh = q.shape[1]
    scores = torch.einsum("nhd,nkd->nhk", q.float(), kv.float()) * scale
    m = torch.maximum(scores.max(-1, keepdim=True).values, sink.view(1, hh, 1))
    e = (scores - m).exp()
    denom = e.sum(-1, keepdim=True) + (sink.view(1, hh, 1) - m).exp()
    return torch.einsum("nhk,nkd->nhd", e / denom, kv.float())


def _ora_sparse_bf16(q, kv, sink, scale):                  # BF16 primitive replica (F4 authoritative)
    hh = q.shape[1]
    qb, kvb = q.bfloat16().float(), kv.bfloat16().float()
    scores = torch.einsum("nhd,nkd->nhk", qb, kvb) * scale
    m = torch.maximum(scores.max(-1, keepdim=True).values, sink.view(1, hh, 1))
    e = (scores - m).exp()
    denom = e.sum(-1, keepdim=True) + (sink.view(1, hh, 1) - m).exp()
    w = e.bfloat16().float()                                # bf16 cast of unnormalized exp before value GEMM
    return (torch.einsum("nhk,nkd->nhd", w, kvb) / denom).bfloat16().float()


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
    ref_fp32 = _ora_sparse(qc, kvc, sink, scale)
    ref_bf16 = _ora_sparse_bf16(qc, kvc, sink, scale)

    c32 = metrics(o_gpu, ref_fp32)
    c16 = metrics(o_gpu, ref_bf16)
    print(f"N={N:>3}  fp32-oracle vs GPU: cos={c32[0]:.6f} mae={c32[1]:.3e} ratio={c32[2]:.5f}   "
          f"bf16-replica vs GPU: cos={c16[0]:.6f} mae={c16[1]:.3e} ratio={c16[2]:.5f}")
    return {"N": N, "scale": scale, "q": qc.cpu(), "kv": kvc.cpu(), "sink": sink.cpu(),
            "gpu_out": o_gpu.cpu(), "fp32_vs_gpu": c32, "bf16_vs_gpu": c16}


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


if __name__ == "__main__":
    main()
