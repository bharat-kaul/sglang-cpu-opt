# Roofline target vs measured — DeepSeek-Flash-v4.1

**Node** `gnr_6980p (FULL NODE, 2 socket, TP=2)` · **fp8** · **decode** · batch 32 · unit `tok/s`

**Model-level:** roofline target **220.0 tok/s** · measured **PENDING** (target published; measured to follow)

Per-op, ranked by recoverable end-to-end fraction (shortfall × phase share):

| op | phase share % | regime | roofline (tok/s) | measured (tok/s) | efficiency | recoverable % |
|----|---------------|--------|-------------------|-------------------|------------|---------------|
| moe_grouped_gemm (routed experts) | 68.0 | memory | 240.0 | PENDING | PENDING | 68.0 |
| mla_attention_core | 12.0 | memory | 1500.0 | PENDING | PENDING | 12.0 |
| dense_proj (q/kv/o) | 8.0 | memory | 2000.0 | PENDING | PENDING | 8.0 |
| dsa_indexer | 5.0 | compute | 4000.0 | PENDING | PENDING | 5.0 |
| norm/rope/act | 4.0 | memory | 6000.0 | PENDING | PENDING | 4.0 |
| lm_head + embed | 3.0 | memory | 3000.0 | PENDING | PENDING | 3.0 |

![roofline vs measured](./deepseek_v4_flash_roofline.svg)

> Bars: roofline-achievable (target) vs measured per op. A tall gap on a high-share op is the top optimization RoI. `PENDING` = target published; measured fills in when the kernel runs.
