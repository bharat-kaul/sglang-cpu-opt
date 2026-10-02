# Roofline target vs measured — DeepSeek-V4-Pro

> **Status: analytical target, reframed (2026-10-01).** DeepSeek-V4-Pro (1.6T) is the *same* DSv4
> architecture as Flash, so it **reuses Flash's authored CPU kernels as donors** — enabling it is
> *wiring + validation*, not new-kernel authoring. It is **too large for one GNR node at tp=1**:
> ~800 GB native MXFP4 exceeds a single 256 GB NUMA domain (and the 768 GB socket), so it can only run
> **TP≥2** — the decode-hostile config (measured 2.8× slower on Flash). Its realistic home is a
> **multi-socket EMR cluster** (Kimi-K3-style TP/EP/PP). The GNR-TP4 numbers below are an earlier
> analytical estimate, **not a result**; the multi-socket roofline is TBD and no single-node Pro run is
> claimed.

**Node** `gnr_6980p (SNC-on: 6 NUMA domains; TP=4 = 4/6 domains; effective 841/1261 GB/s)` · **bf16 (fp4-storage, dequant-in-forward)** · **decode** · batch 32 · unit `tok/s` · **1.6T total / 49B active (MoE)**

**Ceiling provenance:** measured by uPP → `tools/uarch_perf_probe/samples/gnr_pcl-gnrap01_demo.json (uPP: 6 SNC domains, 226 GB/s/domain, remote/local 0.61 — corroborates the SNC-corrected ceiling)`

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
