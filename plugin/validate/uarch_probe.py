#!/usr/bin/env python3
"""uArch perf probe — MEASURE per-domain memory BW and bf16 AMX GEMM peak on the
target node, pinned to ONE NUMA domain. Replaces the injected priors in the platform
spec with measured machine_constants. Run under: srun -w <node> numactl --cpunodebind=0
--membind=0 python uarch_probe.py (OMP threads = one socket).
"""
import json
import os
import sys
import time

import torch

THREADS = int(os.environ.get("PROBE_THREADS", "64"))
torch.set_num_threads(THREADS)


def measure_bw():
    # STREAM-Triad style: a = b + s*c on fp32 arrays >> LLC (640 MB). 3 array touches/elem.
    n = 512 * 1024 * 1024  # 512M fp32 = 2 GB per array
    b = torch.ones(n, dtype=torch.float32)
    c = torch.full((n,), 2.0, dtype=torch.float32)
    a = torch.empty_like(b)
    best = 0.0
    for _ in range(15):
        t0 = time.perf_counter()
        torch.add(b, c, alpha=3.0, out=a)  # a = b + 3*c
        dt = time.perf_counter() - t0
        gbps = 3 * n * 4 / dt / 1e9  # 2 read + 1 write
        best = max(best, gbps)
    return best


def measure_gemm_bf16():
    best = 0.0
    for S in (4096, 8192):
        A = torch.randn(S, S, dtype=torch.bfloat16)
        B = torch.randn(S, S, dtype=torch.bfloat16)
        for _ in range(3):
            torch.matmul(A, B)  # warm
        for _ in range(8):
            t0 = time.perf_counter()
            torch.matmul(A, B)
            dt = time.perf_counter() - t0
            tflops = 2 * S ** 3 / dt / 1e12
            best = max(best, tflops)
    return best


if __name__ == "__main__":
    print(f"threads={THREADS}  numa_bind={os.environ.get('NUMA_NODE','?')}")
    bw = measure_bw()
    gf = measure_gemm_bf16()
    print(json.dumps({"mem_bw_gbps_measured": round(bw, 1),
                      "amx_bf16_tflops_measured": round(gf, 2)}, indent=2))
