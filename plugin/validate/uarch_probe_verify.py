#!/usr/bin/env python3
"""Reverify: NUMA domains/socket + bf16 AMX GEMM peak (careful sweep). Confirms whether
torch.matmul is hitting AMX and finds the achievable plateau vs theoretical."""
import json
import os
import subprocess
import time

import torch

T = int(os.environ.get("PROBE_THREADS", "64"))
torch.set_num_threads(T)

print("=== topology ===")
print(subprocess.run(["bash", "-lc", "lscpu | grep -iE 'NUMA node\\(s\\)|Socket|Core\\(s\\) per socket|Model name'"],
                     capture_output=True, text=True).stdout)
print(subprocess.run(["bash", "-lc", "numactl --hardware | grep -E 'available|node [0-9]+ size'"],
                     capture_output=True, text=True).stdout)
print("torch mkldnn:", torch.backends.mkldnn.is_available(), "| threads:", torch.get_num_threads())

# theoretical bf16 AMX peak: TMUL 16x16x32 = 16384 FLOP, ~1 per 16 cyc => 1024 FLOP/cyc/core
# one socket = cores * 1024 * freq. Report at a few plausible AMX clocks.
cores = 64
for ghz in (1.9, 2.3, 2.9):
    print(f"  theoretical bf16 peak @ {ghz}GHz, {cores}c, 1024 FLOP/cyc/core = {cores*1024*ghz:.0f} GFLOP/s")

print("\n=== bf16 GEMM sweep (take plateau) ===")
best = 0.0
for S in (2048, 4096, 8192, 12288, 16384):
    A = torch.randn(S, S, dtype=torch.bfloat16)
    B = torch.randn(S, S, dtype=torch.bfloat16)
    for _ in range(3):
        torch.matmul(A, B)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        torch.matmul(A, B)
        ts.append(time.perf_counter() - t0)
    dt = min(ts)
    tf = 2 * S ** 3 / dt / 1e12
    best = max(best, tf)
    print(f"  S={S:5d}  best {tf:6.1f} TFLOP/s   (median {2*S**3/ (sorted(ts)[len(ts)//2]) /1e12:6.1f})")
print(f"\nPLATEAU bf16 AMX peak ~= {best:.1f} TFLOP/s")
print(json.dumps({"amx_bf16_tflops_measured": round(best, 2)}))
