#!/usr/bin/env python3
"""Load a PRIVATE sgl_kernel common_ops.so (no venv sgl_kernel import) and run the MXFP4 MoE.
Seeded inputs so gate-off vs gate-on produce identical inputs -> bit-parity diff. Prints ms + output
norm; saves the output tensor to OUT for cross-run parity. Env: SO_PATH, E, M, HIDDEN, INTER, TOPK,
ITERS, THREADS, SEED, OUT. The optimization gate (INTEL_CPU_DSV4_MOE_DBLBUF) is read by the C++ kernel."""
import os
import time

import torch

MXFP4 = 4  # CPUQuantMethod.MXFP4 (hardcoded to avoid importing sgl_kernel)

so = os.environ["SO_PATH"]
torch.ops.load_library(so)  # registers torch.ops.sgl_kernel.* from the PRIVATE build only

E = int(os.environ.get("E", 8))
M = int(os.environ.get("M", 32))
hidden = int(os.environ.get("HIDDEN", 4096))
inter = int(os.environ.get("INTER", 2048))
topk = int(os.environ.get("TOPK", 6))
iters = int(os.environ.get("ITERS", 20))
threads = int(os.environ.get("THREADS", os.cpu_count() or 64))
seed = int(os.environ.get("SEED", 0))
out = os.environ.get("OUT", "")
torch.set_num_threads(threads)
torch.manual_seed(seed)

group = 32
N2 = 2 * inter


def _rand_mxfp4(O, K):
    packed = torch.randint(0, 256, (O, K // 2), dtype=torch.uint8)
    e8m0 = torch.randint(125, 130, (O, K // group), dtype=torch.uint8)
    return packed, e8m0


w13_u8, w13s = zip(*[_rand_mxfp4(N2, hidden) for _ in range(E)])
w2_u8, w2s = zip(*[_rand_mxfp4(hidden, inter) for _ in range(E)])
w13p = torch.ops.sgl_kernel.convert_weight_packed(torch.stack(w13_u8).contiguous())
w2p = torch.ops.sgl_kernel.convert_weight_packed(torch.stack(w2_u8).contiguous())
w13sp = torch.ops.sgl_kernel.convert_scale_packed(torch.stack(w13s).contiguous())
w2sp = torch.ops.sgl_kernel.convert_scale_packed(torch.stack(w2s).contiguous())
x = torch.randn(M, hidden, dtype=torch.bfloat16)
topk_id = torch.stack([torch.randperm(E)[:topk] for _ in range(M)]).to(torch.int32)
topk_w = torch.rand(M, topk, dtype=torch.float32)


def run():
    return torch.ops.sgl_kernel.fused_experts_cpu(
        x, w13p, w2p, topk_w, topk_id, False, MXFP4,
        w13sp, w2sp, None, None, None, None, None, None, None, True, "silu",
    )


y = run()
y = run()  # warm
t0 = time.perf_counter()
for _ in range(iters):
    y = run()
ms = (time.perf_counter() - t0) / iters * 1e3
gate = os.environ.get("INTEL_CPU_DSV4_MOE_DBLBUF", "0")
print(f"PRIVATE-KERNEL  E={E} M={M} topk={topk} threads={threads} gate(dblbuf)={gate}  "
      f"ms={ms:.3f}  out_shape={tuple(y.shape)}  out_norm={y.float().norm().item():.6f}  "
      f"out_sum={y.float().sum().item():.4f}")
if out:
    torch.save(y.float().contiguous(), out)
    print(f"saved output -> {out}")
