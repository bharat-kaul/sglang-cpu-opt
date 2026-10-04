"""DSV4 torch-bucket SIZING pass — separate dispatch-overhead vs fp32-compute vs thread-oversubscription.

Micro-benchmarks the REAL DSA torch ops (compressor softmax-pool, lightning indexer logits+topk,
fp32 MLA attend) and a representative MHC sinkhorn at config-accurate batch-1 DECODE shapes. For each op:
  (1) torch.profiler -> aten dispatch COUNT/call + self CPU us  -> is it dispatch-bound?
  (2) thread sweep {1,8,16,32,64,128}                           -> is it thread-OVERSUBSCRIBED (small tensors)?
  (3) context-length sweep S                                    -> does time scale with FLOPs (compute-bound)?
Verdict per op routes to the right lever: dispatch->fusion, compute->bf16/AMX kernel, oversub->thread-cap.
Diagnostic only; not shipped.
"""
from __future__ import annotations
import os, sys, time, json
import torch

sys.path.insert(0, "/data/nfs_home/bkaul/cpopt-bwprobe/plugin/intel_cpu_models")
from dsa_compressor_cpu import compress_softmax_pool
from dsa_indexer_cpu import indexer_logits, indexer_topk

torch.manual_seed(0)
# config-accurate DSV4-Flash decode dims
H_IDX, D_IDX, TOPK = 64, 128, 512
RATIO, D_CMP = 128, 128
H_ATT, D_ATT = 64, 128
SINK_ITERS, N_HASH = 20, 3
ITERS = int(os.environ.get("ITERS", "50"))
S_LIST = [int(x) for x in os.environ.get("S_LIST", "512,2048,8192").split(",")]
THREADS = [int(x) for x in os.environ.get("THREADS", "1,8,16,32,64,128").split(",")]


def _attend(q, k, v):  # fp32 MLA attend over selected keys: q[H,D] k[S,D] v[S,D]
    a = torch.softmax((q @ k.t()) * (q.shape[-1] ** -0.5), dim=-1)
    return a @ v


def _sinkhorn(logits, iters):  # representative log-domain row/col normalization (hash clustering)
    for _ in range(iters):
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=0, keepdim=True)
    return logits


def make(op, S):
    if op == "indexer":
        q = torch.randn(1, H_IDX, D_IDX); kv = torch.randn(1, S, D_IDX); w = torch.randn(1, H_IDX)
        return lambda: indexer_topk(indexer_logits(q, kv, w), TOPK)
    if op == "compressor":
        N = max(1, S // RATIO); kv = torch.randn(N, RATIO, D_CMP); sc = torch.randn(N, RATIO, D_CMP); ape = torch.randn(RATIO, D_CMP)
        return lambda: compress_softmax_pool(kv, sc, ape)
    if op == "attend":
        q = torch.randn(H_ATT, D_ATT); k = torch.randn(min(TOPK, S), D_ATT); v = torch.randn(min(TOPK, S), D_ATT)
        return lambda: _attend(q, k, v)
    if op == "sinkhorn":
        C = 128; lg = torch.randn(N_HASH, C)
        return lambda: _sinkhorn(lg, SINK_ITERS)
    raise ValueError(op)


def timeit(fn, iters):
    for _ in range(5): fn()
    t = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t) / iters * 1e6  # us/call


def dispatch_count(fn):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(5): fn()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(ITERS): fn()
    ka = prof.key_averages()
    n_aten = sum(e.count for e in ka if e.key.startswith("aten::"))
    self_us = sum(e.self_cpu_time_total for e in ka)
    return n_aten / ITERS, self_us / ITERS


print(f"=== DSV4 torch sizing (ITERS={ITERS}) H_idx={H_IDX} D={D_IDX} topk={TOPK} ratio={RATIO} ===")
Smid = S_LIST[len(S_LIST) // 2]
for op in ["indexer", "compressor", "attend", "sinkhorn"]:
    torch.set_num_threads(THREADS[0])
    disp, self_us = dispatch_count(make(op, Smid))
    # thread sweep at mid S
    tsweep = {}
    for t in THREADS:
        torch.set_num_threads(t)
        tsweep[t] = timeit(make(op, Smid), ITERS)
    torch.set_num_threads(THREADS[0])
    # context sweep single-thread (compute-scaling)
    csweep = {S: timeit(make(op, S), ITERS) for S in S_LIST}
    best_t = min(tsweep, key=tsweep.get); worst = max(tsweep.values()); best = tsweep[best_t]
    oversub = worst / best if best > 0 else 0
    scale = csweep[S_LIST[-1]] / csweep[S_LIST[0]] if csweep[S_LIST[0]] > 0 else 0
    print(f"\n[{op}] dispatches/call={disp:.1f}  1-thread={tsweep[THREADS[0]]:.1f}us")
    print(f"  thread-sweep us: " + " ".join(f"{t}t={tsweep[t]:.1f}" for t in THREADS) + f"  -> best@{best_t}t, oversub(worst/best)={oversub:.1f}x")
    print(f"  ctx-sweep us:    " + " ".join(f"S{S}={csweep[S]:.1f}" for S in S_LIST) + f"  -> x{scale:.1f} over {S_LIST[0]}->{S_LIST[-1]}")
    # crude verdict
    v = []
    if disp >= 15: v.append("DISPATCH-heavy(fuse)")
    if oversub >= 1.5: v.append(f"OVERSUBSCRIBED(cap~{best_t}t)")
    if scale >= 3: v.append("COMPUTE-scales(bf16/AMX)")
    print(f"  VERDICT: {', '.join(v) if v else 'flat/small — low upside'}")
print("\n[done]")
