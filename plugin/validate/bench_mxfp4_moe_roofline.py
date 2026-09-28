"""Roofline microbench for the native MXFP4 W4A16 CPU MoE kernel (the dominant decode op).

Isolation perf gate (independent of full-model accuracy): times
`fused_experts_cpu(CPUQuantMethod.MXFP4)` at the REAL DeepSeek-V4-Flash MoE shape across a
thread sweep, and backs out the EFFECTIVE bytes/s. Decode is weight-streaming-bound, so the
ceiling is memory bandwidth; comparing measured effective BW against (a) top-k-expert bytes and
(b) all-expert bytes reveals whether decode streams only the routed experts or the whole table
(a big potential win) and how far the kernel is from the BW roofline.

Run on the TARGET silicon for real numbers (GNR); EMR/SPR gives a directional read (the
top-k-vs-all-experts ratio is node-independent). Perf numbers are labeled by node.

  BW_GBPS=<one-domain GB/s> M=1 THREADS=8,16,32,43 \
  PYTHONPATH=<plugin> srun ... python plugin/validate/bench_mxfp4_moe_roofline.py
"""
import os
import time

import torch


def _rand_mxfp4(O, K, group=32):
    packed = torch.randint(0, 256, (O, K // 2), dtype=torch.uint8)
    e8m0 = torch.randint(125, 130, (O, K // group), dtype=torch.uint8)  # scales ~1
    return packed, e8m0


def main() -> None:
    torch.manual_seed(0)
    try:
        import sgl_kernel  # noqa: F401  — registers torch.ops.sgl_kernel.*
        from sglang.srt.layers.amx_utils import CPUQuantMethod
    except Exception as e:  # noqa: BLE001
        print(f"MXFP4 MoE roofline: SKIP ({e})")
        return

    # DeepSeek-V4-Flash MoE shape
    E = int(os.environ.get("E", 256))
    topk = int(os.environ.get("TOPK", 6))
    hidden = int(os.environ.get("HIDDEN", 4096))
    inter = int(os.environ.get("INTER", 2048))
    M = int(os.environ.get("M", 1))                 # 1 = decode, e.g. 64 = prefill tile
    iters = int(os.environ.get("ITERS", 20))
    bw_gbps = float(os.environ.get("BW_GBPS", 226.0))   # one GNR SNC domain (uPP); override per node
    node = os.environ.get("NODE_LABEL", os.environ.get("SLURMD_NODENAME", "unknown"))
    threads_env = os.environ.get("THREADS", "8,16,32,43")
    thread_list = [int(t) for t in threads_env.split(",") if t]

    N2 = 2 * inter
    x = torch.randn(M, hidden, dtype=torch.bfloat16)
    w13_u8, w13s = zip(*[_rand_mxfp4(N2, hidden) for _ in range(E)])
    w2_u8, w2s = zip(*[_rand_mxfp4(hidden, inter) for _ in range(E)])
    w13p = torch.ops.sgl_kernel.convert_weight_packed(torch.stack(w13_u8).contiguous())
    w2p = torch.ops.sgl_kernel.convert_weight_packed(torch.stack(w2_u8).contiguous())
    w13sp = torch.ops.sgl_kernel.convert_scale_packed(torch.stack(w13s).contiguous())
    w2sp = torch.ops.sgl_kernel.convert_scale_packed(torch.stack(w2s).contiguous())

    topk_id = torch.stack([torch.randperm(E)[:topk] for _ in range(M)]).to(torch.int32)
    topk_w = torch.rand(M, topk, dtype=torch.float32)

    # Bytes: fp4 weights = 0.5 B/elem; e8m0 scale = 1 B / 32 elem. Per expert:
    per_expert_w = (N2 * hidden + hidden * inter) * 0.5
    per_expert_s = (N2 * hidden + hidden * inter) / 32.0
    per_expert = per_expert_w + per_expert_s
    bytes_topk = per_expert * topk * M      # if only routed experts are streamed (M tokens)
    bytes_all = per_expert * E              # if the whole table is streamed once

    def run():
        return torch.ops.sgl_kernel.fused_experts_cpu(
            x, w13p, w2p, topk_w, topk_id, False, CPUQuantMethod.MXFP4,
            w13sp, w2sp, None, None, None, None, None, None, None, True, "silu",
        )

    print(f"# MXFP4 MoE roofline | node={node} E={E} topk={topk} hidden={hidden} "
          f"inter={inter} M={M} | BW_ceiling={bw_gbps:.0f} GB/s")
    print(f"# per-expert={per_expert/1e6:.2f} MB  topk-bytes={bytes_topk/1e6:.1f} MB  "
          f"all-expert-bytes={bytes_all/1e6:.1f} MB")
    print(f"# {'threads':>7} {'ms':>9} {'GB/s(topk)':>11} {'GB/s(all)':>10} "
          f"{'eff_topk':>9} {'eff_all':>8}")
    for nth in thread_list:
        torch.set_num_threads(nth)
        run(); run()  # warm
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        ms = (time.perf_counter() - t0) / iters * 1e3
        gbps_topk = bytes_topk / (ms / 1e3) / 1e9
        gbps_all = bytes_all / (ms / 1e3) / 1e9
        print(f"  {nth:>7} {ms:>9.3f} {gbps_topk:>11.1f} {gbps_all:>10.1f} "
              f"{gbps_topk / bw_gbps:>9.2f} {gbps_all / bw_gbps:>8.2f}")
    print("# eff_topk≈1 -> streaming only routed experts (at BW roofline); "
          "eff_topk<<1 while eff_all≈1 -> streaming ALL experts (optimization opportunity).")


if __name__ == "__main__":
    main()
