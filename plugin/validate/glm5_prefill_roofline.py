#!/usr/bin/env python3
"""GLM-5.3 Flash CPU prefill roofline: per-op GEMM FLOPs (8-layer proxy, M=256) vs the
MEASURED GNR AMX bf16 GEMM ceiling. Emits results/glm5_flash_prefill_roofline.json.

The optimized prefill is compute-shaped (AMX matmuls) but runs at a tiny fraction of
peak because the effective matmuls are SMALL (MoE routes ~M*topk/E tokens/expert; KDA
uses chunk=16) -> the roofline gap is small-matmul / operating-point inefficiency, the
same lever DSv4 identifies (batching), not a missing kernel. This quantifies it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

# --- GLM-5.3 Flash dims (config.json text_config) ---
HID = 4096
KDA_H, KDA_D = 64, 128           # linear_attn_config num_heads / head_dim
PROJ = KDA_H * KDA_D             # 8192
MLA_H, QK_NOPE, V_HD = 64, 256, 256
Q_LORA, KV_LORA = 1536, 512
DENSE_I = 12288
MOE_I, N_EXP, TOPK, N_SHARED = 2048, 288, 8, 1
M = 256                          # prefill tokens (proxy)
# 8-layer proxy composition (L0-7): KDA L0,1,2,4,5,6 (6); MLA L3,7 (2);
# dense MLP L0,1,2 (3); MoE L3-7 (5).
N_KDA, N_MLA, N_DENSE, N_MOE = 6, 2, 3, 5


def gemm(m, n, k):
    return 2.0 * m * n * k


def kda_flops():
    proj = (
        gemm(M, 3 * PROJ, HID)                 # qkv_proj
        + gemm(M, KDA_H, HID)                   # b_proj
        + gemm(M, KDA_D, HID) + gemm(M, PROJ, KDA_D)   # f_a, f_b
        + gemm(M, KDA_D, HID) + gemm(M, PROJ, KDA_D)   # g_a, g_b
        + gemm(M, HID, PROJ)                    # o_proj
    )
    # chunked gated-delta recurrence: 64 heads, K=V=128, 16 chunks of 16.
    C, NC = 16, M // 16
    per_ch_head = (
        gemm(C, C, KDA_D)          # A_kk
        + gemm(C, C, KDA_D)        # A_qk
        + C * C * KDA_D            # tri-solve (~half a gemm)
        + 2 * gemm(C, V_HD if False else KDA_D, KDA_D)  # Su, So (@S)
        + gemm(C, KDA_D, C)        # O cross
        + gemm(KDA_D, KDA_D, C)    # state KD^T@Vp
    )
    recur = per_ch_head * NC * KDA_H
    return proj + recur


def mla_flops():
    return (
        gemm(M, Q_LORA, HID) + gemm(M, MLA_H * QK_NOPE, Q_LORA)   # q_a, q_b
        + gemm(M, KV_LORA, HID) + gemm(M, MLA_H * (QK_NOPE + V_HD), KV_LORA)  # kv_a, kv_b
        + gemm(M, HID, MLA_H * V_HD)                              # o_proj
        + 2 * gemm(M, M, MLA_H * QK_NOPE)                         # QK^T + AV (dense, NoPE)
    )


def dense_flops():
    return 2 * gemm(M, DENSE_I, HID) + gemm(M, HID, DENSE_I)      # gate+up, down


def moe_flops():
    # routed: M*topk token-expert pairs; + shared expert on all M. Each expert = gate+up+down.
    per_tok_expert = 2 * gemm(1, MOE_I, HID) + gemm(1, HID, MOE_I)
    return (M * TOPK + M * N_SHARED) * per_tok_expert


def measure_amx_bf16_peak():
    torch.set_num_threads(torch.get_num_threads())
    best = 0.0
    for (m, n, k) in [(4096, 4096, 4096), (2048, 8192, 4096), (8192, 4096, 4096)]:
        a = torch.randn(m, k, dtype=torch.bfloat16)
        b = torch.randn(k, n, dtype=torch.bfloat16)
        for _ in range(3):
            torch.matmul(a, b)
        t = time.perf_counter()
        it = 5
        for _ in range(it):
            torch.matmul(a, b)
        dt = (time.perf_counter() - t) / it
        gf = gemm(m, n, k) / dt / 1e9
        best = max(best, gf)
    return best  # GFLOP/s


if __name__ == "__main__":
    ops = {
        "kda": (N_KDA, kda_flops()),
        "mla": (N_MLA, mla_flops()),
        "dense": (N_DENSE, dense_flops()),
        "moe": (N_MOE, moe_flops()),
    }
    total = sum(n * f for n, f in ops.values())
    peak = measure_amx_bf16_peak()
    roof_s = total / (peak * 1e9)
    out = {
        "_comment": "GLM-5.3 Flash CPU prefill roofline (8-layer proxy, M=256, GNR). "
        "Baseline/optimized are MEASURED layer.total.pf (ledger). Roofline = total GEMM "
        "FLOPs / measured GNR AMX bf16 GEMM ceiling. The large roofline gap is small-matmul "
        "inefficiency (MoE ~M*topk/E tokens/expert; KDA chunk=16) -> operating-point/batching "
        "lever (cf DSv4), not a missing kernel.",
        "node": "GNR (gnrap, 256 threads)",
        "threads": torch.get_num_threads(),
        "M_tokens": M,
        "amx_bf16_peak_gflops": round(peak, 1),
        "total_prefill_gflops": round(total / 1e9, 1),
        "per_op_gflops": {k: round(n * f / 1e9, 1) for k, (n, f) in ops.items()},
        "baseline_s": 55.535,
        "optimized_s": 9.791,
        "roofline_s": round(roof_s, 4),
        "speedup_measured": round(55.535 / 9.791, 2),
        "optimized_pct_of_roofline": round(100 * roof_s / 9.791, 2),
    }
    p = Path(__file__).parent / "results" / "glm5_flash_prefill_roofline.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {p}")
