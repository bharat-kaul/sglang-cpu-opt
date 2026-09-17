# Roofline target vs measured — DeepSeek-V4-Flash

**Node** `gnr_6980p (SNC-on: 6 NUMA domains; heads not div by 6 -> TP=4 = 4/6 domains; effective 841/1261 GB/s)` · **fp8** · **decode** · batch 32 · unit `tok/s` · **284B total / 13B active (MoE)**

**Ceiling provenance:** measured by uPP → `tools/uarch_perf_probe/samples/gnr_pcl-gnrap01_demo.json (uPP: 6 SNC domains, 226 GB/s/domain, remote/local 0.61 — corroborates the SNC-corrected ceiling)`

**Model-level:** roofline target **146.7 tok/s** · measured **PENDING** (target published; measured to follow)

Per-op, ranked by recoverable end-to-end fraction (shortfall × phase share):

| op | phase share % | regime | roofline (tok/s) | measured (tok/s) | efficiency | recoverable % |
|----|---------------|--------|-------------------|-------------------|------------|---------------|
| moe_grouped_gemm (routed experts) | 68.0 | memory | 160.0 | PENDING | PENDING | 68.0 |
| mla_attention_core | 12.0 | memory | 1000.0 | PENDING | PENDING | 12.0 |
| dense_proj (q/kv/o) | 8.0 | memory | 1333.3 | PENDING | PENDING | 8.0 |
| dsa_indexer | 5.0 | compute | 2666.7 | PENDING | PENDING | 5.0 |
| norm/rope/act | 4.0 | memory | 4000.0 | PENDING | PENDING | 4.0 |
| lm_head + embed | 3.0 | memory | 2000.0 | PENDING | PENDING | 3.0 |

![roofline vs measured](./deepseek_v4_flash_roofline.png)

> Bars: roofline-achievable (target) vs measured per op. A tall gap on a high-share op is the top optimization RoI. `PENDING` = target published; measured fills in when the kernel runs.
