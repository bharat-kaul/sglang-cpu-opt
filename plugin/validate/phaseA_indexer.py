#!/usr/bin/env python3
"""Phase-A op #1: DSA indexer logits — correctness (cosine vs reference) + roofline-vs-measured.
Authored fresh; cross-checked against the reference semantics in dsa_indexer_cpu.py.
Run on EMR: srun -w <node> numactl --cpunodebind=0 --membind=0 python phaseA_indexer.py
"""
import json
import os
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)  # plugin/
import sys
sys.path.insert(0, os.path.join(ROOT, "intel_cpu_models"))
from dsa_indexer_cpu import indexer_logits as ref_logits  # the reference port

PLAT = json.load(open(os.path.join(HERE, "platforms", "emr.json")))
BW = PLAT["mem_bw_gbps"] * 1e9
PEAK = PLAT["amx_bf16_tflops"] * 1e12
THREADS = int(os.environ.get("PROBE_THREADS", "64"))
torch.set_num_threads(THREADS)

mod = load(name="dsa_idx_pilot",
           sources=[os.path.join(ROOT, "kernels", "dsa_pilot", "indexer_logits.cpp")],
           extra_cflags=["-O3", "-march=native", "-fopenmp", "-mavx512bf16", "-mamx-tile",
                         "-funsafe-math-optimizations"],
           verbose=False)

H, D, S, K = 64, 128, 4096, 512  # index_n_heads, index_head_dim, context, index_topk
AB = 2.0


def roofline_bytes(N):
    # fused: read kv (N*S*D bf16) + q (N*H*D bf16); scores stay L2-resident; write logits N*S fp32
    return (N * S * D * AB + N * H * D * AB + N * S * 4)


def roofline_flops(N):
    return 2 * N * H * S * D + 3 * N * H * S  # gemm + epilogue


def bench(fn, *a, iters=20):
    for _ in range(3):
        fn(*a)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(*a); ts.append(time.perf_counter() - t0)
    return min(ts)


print(f"platform={PLAT['name']} BW={BW/1e9:.0f}GB/s PEAK={PEAK/1e12:.1f}TF ridge={PEAK/BW:.0f} threads={THREADS}")
print(f"{'N':>4} {'in':>5} {'cos':>10} {'ref ms':>9} {'cpp ms':>9} {'speedup':>8} {'ceil ms':>9} {'cpp/ceil':>9} {'bound':>8}")
for N in (1, 8, 16, 32, 64):
    q0 = torch.randn(N, H, D)
    kv0 = torch.randn(N, S, D)
    w = torch.rand(N, H)
    r = ref_logits(q0, kv0, w)
    byts, fl = roofline_bytes(N), roofline_flops(N)
    ceil = max(byts / BW, fl / PEAK)
    bound = "compute" if fl / byts > PEAK / BW else "BW"
    for tag, q, kv in (("fp32", q0, kv0), ("bf16", q0.bfloat16(), kv0.bfloat16())):
        c = mod.indexer_logits_fused(q, kv, w)
        cos = torch.nn.functional.cosine_similarity(r.flatten().float(), c.flatten().float(), dim=0).item()
        t_ref = bench(ref_logits, q0, kv0, w)
        t_cpp = bench(mod.indexer_logits_fused, q, kv, w)
        print(f"{N:>4} {tag:>5} {cos:>10.6f} {t_ref*1e3:>9.3f} {t_cpp*1e3:>9.3f} {t_ref/t_cpp:>7.2f}x "
              f"{ceil*1e3:>9.3f} {t_cpp/ceil:>8.2f}x {bound:>8}")

