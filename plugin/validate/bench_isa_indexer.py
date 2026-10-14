#!/usr/bin/env python3
"""ISA-path triage for the indexer score GEMM (test of the 'arch-invariant structure, per-target MEASURED
compute path' rule). The indexer's AMX brgemm runs at ~5.4 TF (4.3% of AMX peak, ~69% of AVX-512-fp32 peak)
on the narrow K=128/N=64 shape -> hypothesis: AMX underutilizes and an AVX-512 path is competitive/wins.

Measures the real score-GEMM shape kv[M,S,D] @ qT[M,D,H] -> [M,S,H] at the indexer dims, STREAMING, three ways:
  bf16 at::bmm  -> dispatches AMX (oneDNN) on EMR/GNR
  fp32 at::bmm  -> dispatches AVX-512 fp32 (oneDNN)
  cpublas brgemm bf16 (the kernel's actual primitive, per S-tile) -> AMX
Reports us + effective TFLOP/s per M. Set ONEDNN_VERBOSE=1 to confirm the ISA that ran. EMR (has AMX + avx512_bf16).
"""
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
H, D, S = 64, 128, 1024                       # indexer score-GEMM dims: scores[S,H] = kv[S,D] @ qT[D,H]
MS = [1, 2, 4, 8, 16, 32, 64]
AMX_TF, FP32_TF = 124.5e12, 7.78e12
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


def main():
    il = load(name="isa_il", sources=[os.path.join(KD, "indexer_logits.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# ISA triage HEAD={sha} threads={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"node={os.uname().nodename}")
    print(f"# score GEMM: kv[M,{S},{D}] @ qT[M,{D},{H}] -> [M,{S},{H}]; FLOPs=2*M*S*H*D")
    print(f"{'M':>4} {'bf16_AMX':>9} {'fp32_AVX':>9} {'brgemm':>8} | {'AMX_TF':>7} {'AVX_TF':>7} {'AMX/AVX':>7} {'bf16/fp32':>9}")
    for M in MS:
        kv = torch.randn(M, S, D); q = torch.randn(M, H, D); w = torch.randn(M, H)
        kvb, qTb = kv.to(torch.bfloat16), q.transpose(1, 2).to(torch.bfloat16)   # [M,S,D],[M,D,H]
        kvf, qTf = kv, q.transpose(1, 2).contiguous()
        t_bf16 = tstream(lambda: torch.bmm(kvb, qTb))                            # -> AMX
        t_fp32 = tstream(lambda: torch.bmm(kvf, qTf))                            # -> AVX-512 fp32
        os.environ["INDEXER_SKIP_EPILOGUE"] = "1"
        t_brg = tstream(lambda: il.indexer_logits(q, kv, w))                     # kernel brgemm (AMX) + read
        os.environ.pop("INDEXER_SKIP_EPILOGUE", None)
        flops = 2.0 * M * S * H * D
        amx_tf = flops / (t_bf16 * 1e-6) / 1e12
        avx_tf = flops / (t_fp32 * 1e-6) / 1e12
        print(f"{M:>4} {t_bf16:>9.1f} {t_fp32:>9.1f} {t_brg:>8.1f} | {amx_tf:>7.2f} {avx_tf:>7.2f} "
              f"{t_fp32 / t_bf16:>7.2f} {t_bf16 / t_fp32:>9.2f}")
    print("# AMX_TF/AVX_TF are effective TFLOP/s. If fp32-AVX is within ~2x of bf16-AMX, a bf16-AVX512 (dpbf16,")
    print("# 2x fp32 throughput, same bytes as AMX) path would MATCH/BEAT AMX -> the ISA rule applies to indexer.")


if __name__ == "__main__":
    main()
