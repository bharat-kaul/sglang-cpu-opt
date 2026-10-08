#!/usr/bin/env python3
"""Phase-A op #7 — Lane-A ceiling verification (roofline-vs-measured, no Engine).

The Lane-A ops (MLA proj, MoE expert, lm_head) are DONOR GEMMs (already wired via
torch.matmul -> oneDNN/AMX). This confirms they reach the MACHINE-PEAK roofline on this
platform (or surfaces the gap), per op/per-M, using the measured platform constants.
Weights measured in bf16 (the AMX compute dtype); fp8/fp4 donors stream fewer weight
bytes, so this bf16 BW figure is a conservative floor for the real fp8/fp4 paths.
"""
import json
import os
import time

import torch

_SPEC = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
with open(_SPEC) as f:
    P = json.load(f)
MP = P["machine_peak"]
BW = MP["mem_bw_gbps"] * 1e9
PEAK = MP["amx_bf16_tflops"] * 1e12
RIDGE = PEAK / BW
AB = 2.0  # bf16 bytes

# Donor GEMM shapes (K->N) from the DSv4-Flash config.
SHAPES = [
    ("MLA wqkv_a", 4096, 1536),
    ("MLA wq_b", 1024, 32768),
    ("MLA wo_a", 32768, 1024),
    ("MLA wo_b", 1024, 4096),
    ("DSA indexer wq", 4096, 8192),
    ("MoE expert gate_up", 4096, 4096),
    ("MoE expert down", 2048, 4096),
    ("lm_head", 4096, 129280),
]
Ms = [1, 8, 16, 32, 64]


def bench_mm(A, B, it=20):
    torch.matmul(A, B)
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        torch.matmul(A, B)
        ts.append(time.perf_counter() - t0)
    return min(ts)


print(f"Lane-A ceiling verify  machine peak: {PEAK/1e12:.1f} TF  {BW/1e9:.1f} GB/s  ridge {RIDGE:.0f}")
print(f"{'op':20} {'M':>4} {'TF':>7} {'GB/s':>7} {'bind':>5} {'%ceiling':>9}")
for name, Kd, Nd in SHAPES:
    Bm = torch.randn(Kd, Nd, dtype=torch.bfloat16)
    for M in Ms:
        Am = torch.randn(M, Kd, dtype=torch.bfloat16)
        dt = bench_mm(Am, Bm)
        flop = 2 * M * Kd * Nd
        byts = (Kd * Nd + M * Kd + M * Nd) * AB
        ai = flop / byts
        tf = flop / dt / 1e12
        gbs = byts / dt / 1e9
        if ai > RIDGE:
            bind, pct = "C", tf / (PEAK / 1e12) * 100
        else:
            bind, pct = "B", gbs / (BW / 1e9) * 100
        print(f"{name:20} {M:>4} {tf:>7.1f} {gbs:>7.1f} {bind:>5} {pct:>8.1f}%")
