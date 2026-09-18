"""Isolate the M=1 (decode) dense-GEMM thread behavior on AMX.

The GNR decode profile showed weight_packed_linear / fp8_scaled_mm_cpu at ~0.25s per M=1
call (~3500x above the weight-streaming floor) even on AMX. This sweeps torch thread count
for the bf16 packed GEMM at realistic Flash dense shapes to test whether it is the same
inverse-thread-scaling pathology already found in fused_experts (=> a decode thread cap fix).

Run on a GNR (AMX) node.
"""
import time

import torch

from sglang.srt.layers.amx_utils import amx_process_weight_after_loading


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # ms median


# Realistic Flash dense shapes (K=in, N=out) at decode M=1.
SHAPES = [
    ("q_a  4096x1536", 4096, 1536),
    ("kv_a 4096x576 ", 4096, 576),
    ("gate 4096x2048", 4096, 2048),
    ("down 2048x4096", 2048, 4096),
    ("o_pr 32768x4096", 32768, 4096),
]
THREADS = [1, 2, 4, 8, 16, 32, 42]
M = 1

print(f"AMX weight_packed_linear, M={M}, median ms/call vs threads")
print("shape            " + "".join(f"{t:>8}" for t in THREADS))
for name, K, N in SHAPES:
    w = torch.randn(N, K, dtype=torch.bfloat16)
    wp = amx_process_weight_after_loading(w)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    row = []
    for nt in THREADS:
        torch.set_num_threads(nt)
        t = bench(lambda: torch.ops.sgl_kernel.weight_packed_linear(x, wp, None, True))
        row.append(t)
    best = min(row)
    print(
        f"{name:16s}"
        + "".join(f"{v:>8.3f}" for v in row)
        + f"   best={best:.3f}ms@{THREADS[row.index(best)]}thr"
    )

# Does a per-call set_num_threads toggle (as the MoE cap does) wreck the next GEMM?
print("\nset_num_threads-toggle penalty (steady 32thr vs toggle 4<->32 before each call):")
for name, K, N in SHAPES:
    w = torch.randn(N, K, dtype=torch.bfloat16)
    wp = amx_process_weight_after_loading(w)
    x = torch.randn(M, K, dtype=torch.bfloat16)

    torch.set_num_threads(32)
    t_steady = bench(lambda: torch.ops.sgl_kernel.weight_packed_linear(x, wp, None, True))

    def toggled():
        torch.set_num_threads(4)
        torch.set_num_threads(32)
        return torch.ops.sgl_kernel.weight_packed_linear(x, wp, None, True)

    t_toggle = bench(toggled)
    print(
        f"{name:16s} steady={t_steady:8.3f}ms  toggle={t_toggle:8.3f}ms  "
        f"penalty={t_toggle / max(t_steady, 1e-9):6.1f}x"
    )

