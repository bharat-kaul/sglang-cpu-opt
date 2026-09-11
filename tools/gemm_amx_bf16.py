#!/usr/bin/env python3
"""BF16 AMX GEMM benchmark for Intel Xeon (Granite Rapids+).

torch.matmul on bfloat16 CPU tensors dispatches through oneDNN to the AMX
TDPBF16PS tile kernels (FP32 accumulation) when the CPU advertises amx_bf16.
Emits one JSON line with achieved TFLOP/s and (optionally) FP32 correctness.
"""
import argparse
import json
import time

import torch


def amx_supported() -> bool:
    fn = getattr(torch._C._cpu, "_is_amx_tile_supported", None)
    return bool(fn()) if fn else False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    p.add_argument("--check", action="store_true", help="run FP32 correctness diff")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    torch.manual_seed(0)
    M, N, K = args.M, args.N, args.K
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)

    for _ in range(args.warmup):
        c = torch.matmul(a_bf16, b_bf16)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        c = torch.matmul(a_bf16, b_bf16)
    t1 = time.perf_counter()

    sec = (t1 - t0) / args.iters
    flop = 2.0 * M * N * K
    tflops = flop / sec / 1e12

    out = {
        "tag": args.tag,
        "M": M, "N": N, "K": K,
        "threads": torch.get_num_threads(),
        "amx_supported": amx_supported(),
        "iters": args.iters,
        "sec_per_iter": sec,
        "tflops": tflops,
    }

    if args.check:
        c_ref = torch.matmul(a, b)  # FP32 reference
        c_test = c.to(torch.float32)
        diff = (c_test - c_ref).abs()
        out["mean_rel_err"] = (diff / (c_ref.abs() + 1e-3)).mean().item()
        out["max_abs_err"] = diff.max().item()
        out["has_nan"] = bool(torch.isnan(c_test).any())

    print(json.dumps(out))


if __name__ == "__main__":
    main()
