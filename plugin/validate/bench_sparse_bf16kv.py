#!/usr/bin/env python3
"""Sparse-attend: quantify the bf16-KV (Phase-2 integration) upside. The bf16 kernel lost standalone because the
fp32->bf16 KV CONVERSION cost more than bf16 saved. This isolates that: measure the bf16 path with the KV
PRE-CAST to bf16 OUTSIDE the timed region (= a bf16 KV cache, the GPU scenario). If bf16-KV beats fp32, sparse's
real lever is Phase-2 (bf16 KV cache), not another Phase-1 variant.

Per M:
  t_fp32            shipped fp32 donor (sparse_attend_bestof_hx)
  t_bf16_fp32in     full bf16 (sparse_attend_bf16bmm_hx), KV arrives fp32  -> pays the cast (the rejected path)
  t_bf16_bf16in     full bf16, KV arrives bf16 (cast hoisted out)          -> simulates a bf16 KV cache
  cast_cost         ~ t_bf16_fp32in - t_bf16_bf16in
  cos_bf16          tolerance of bf16 vs fp32 output (normal + sharp dist)
Also prints the fp32 vs bf16 roofline floors. EMR (AMX + avx512_bf16).
"""
import math
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
H, D, K = 64, 512, 512
MS = [1, 2, 4, 8, 16, 32, 64]
BW, AMX_TF, FP32_TF = 358.4e9, 124.5e12, 7.78e12
torch.manual_seed(0)
from torch.utils.cpp_extension import load

_FLUSH = None
_SINK = 0.0


def _flush():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)
    _FLUSH.uniform_(-1.0, 1.0)
    _SINK += float(_FLUSH[::4096].sum())


def _call(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def tstream(fn, it=25):
    fn()
    return st.median([(_flush(), _call(fn))[1] for _ in range(it)]) * 1e6


def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    sp = load(name="skv_sp", sources=[os.path.join(KD, "sparse_attend.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# sparse bf16-KV upside HEAD={sha} threads={torch.get_num_threads()} node={os.uname().nodename}")
    # roofline floors (M=32): FLOPs=4*M*H*K*D; fp32 vs bf16 compute, bytes = kv+q+out
    for M in (1, 32):
        flops = 4.0 * M * H * K * D
        b_fp32 = (M * K * D + M * H * D + M * H * D) * 4
        b_bf16 = (M * K * D + M * H * D + M * H * D) * 2
        fp32_floor = max(flops / FP32_TF, b_fp32 / BW) * 1e6
        bf16_floor = max(flops / AMX_TF, b_bf16 / BW) * 1e6
        print(f"# M={M}: fp32 floor={fp32_floor:.1f}us (compute {flops/FP32_TF*1e6:.1f}) | "
              f"bf16 floor={bf16_floor:.1f}us (compute {flops/AMX_TF*1e6:.1f}, mem {b_bf16/BW*1e6:.1f})")
    print(f"{'M':>4} {'fp32':>8} {'bf16_fp32in':>11} {'bf16_bf16in':>11} {'cast':>7} {'bf16KV/fp32':>11} {'cos':>7}")
    sharp = False
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, K, D); sink = torch.randn(H); scale = D ** -0.5
        if sharp:
            q *= 4.0
        kvb = kv.to(torch.bfloat16).contiguous()                        # bf16 KV cache (cast hoisted OUT of timing)
        o_fp32 = sp.sparse_attend_bestof_hx(q, kv, sink, scale)
        o_bf16 = sp.sparse_attend_bf16bmm_hx(q, kvb, sink, scale)
        t_fp32 = tstream(lambda: sp.sparse_attend_bestof_hx(q, kv, sink, scale))
        t_bf16_fp32in = tstream(lambda: sp.sparse_attend_bf16bmm_hx(q, kv, sink, scale))
        t_bf16_bf16in = tstream(lambda: sp.sparse_attend_bf16bmm_hx(q, kvb, sink, scale))
        cast = t_bf16_fp32in - t_bf16_bf16in
        print(f"{M:>4} {t_fp32:>8.1f} {t_bf16_fp32in:>11.1f} {t_bf16_bf16in:>11.1f} {cast:>7.1f} "
              f"{t_fp32 / t_bf16_bf16in:>10.2f}x {_cos(o_fp32, o_bf16):>7.4f}")
    print("# bf16KV/fp32 > 1 => bf16 with a bf16 KV cache BEATS fp32 => Phase-2 bf16-KV is the real sparse lever.")
    print("# if still <= 1 even with the cast removed => the value GEMM / softmax is the wall, not the conversion.")


if __name__ == "__main__":
    main()
