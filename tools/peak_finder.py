#!/usr/bin/env python3
"""Find best achievable BF16 AMX TFLOP/s on this node across shapes/threads.

Establishes an empirical 'achievable peak' anchor to sit alongside the
theoretical AMX ceiling in the roofline verdict.
"""
import json
import os
import time

import torch

SHAPES = [2048, 4096, 8192, 16384]
ITERS = 30


def bench(m, n, k, iters=ITERS, warmup=8):
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    for _ in range(warmup):
        c = torch.matmul(a, b)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    t1 = time.perf_counter()
    sec = (t1 - t0) / iters
    return 2.0 * m * n * k / sec / 1e12


def main():
    t = int(os.environ.get("PEAK_THREADS", "0"))
    if t > 0:
        torch.set_num_threads(t)
    best = 0.0
    for s in SHAPES:
        tf = bench(s, s, s)
        best = max(best, tf)
        print(json.dumps({"shape": s, "threads": torch.get_num_threads(),
                          "tflops": round(tf, 1)}), flush=True)
    print(json.dumps({"best_achievable_tflops": round(best, 1),
                      "threads": torch.get_num_threads()}), flush=True)


if __name__ == "__main__":
    main()
