# Roofline target vs measured — DeepSeek-V4-Pro

**Node** `gnr_6980p (SNC-on: 6 NUMA domains; TP=4 = 4/6 domains; effective 841/1261 GB/s)` · **bf16 (fp4-storage, dequant-in-forward)** · **decode** · batch 32 · unit `tok/s` · **1.6T total / 49B active (MoE)**

**Model-level:** roofline target **38.7 tok/s** · measured **PENDING** (target published; measured to follow)

Per-op, ranked by recoverable end-to-end fraction (shortfall × phase share):

| op | phase share % | regime | roofline (tok/s) | measured (tok/s) | efficiency | recoverable % |
|----|---------------|--------|-------------------|-------------------|------------|---------------|
| moe_grouped_gemm (routed experts) | 72.0 | memory | 42.0 | PENDING | PENDING | 72.0 |
| mla_attention_core | 10.0 | memory | 260.0 | PENDING | PENDING | 10.0 |
| dense_proj (q/kv/o) | 7.0 | memory | 350.0 | PENDING | PENDING | 7.0 |
| dsa_indexer | 5.0 | compute | 700.0 | PENDING | PENDING | 5.0 |
| norm/rope/act | 3.0 | memory | 1050.0 | PENDING | PENDING | 3.0 |
| lm_head + embed | 3.0 | memory | 530.0 | PENDING | PENDING | 3.0 |

![roofline vs measured](./deepseek_v4_pro_roofline.png)

> Bars: roofline-achievable (target) vs measured per op. A tall gap on a high-share op is the top optimization RoI. `PENDING` = target published; measured fills in when the kernel runs.
