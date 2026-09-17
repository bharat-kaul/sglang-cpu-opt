# Roofline target vs measured — OLMo-2-7B donor kernels (Thesis-1 reuse)

**Node** `gnr_6980p (SINGLE SOCKET, TP=1, 128c NUMA 0-2, batch 8)` · **bf16** · **prefill (GEMM-bound; measured = reused donor kernel; ceiling = bandwidth)** · batch 8 · unit `TF/s` · **7B dense**

**Ceiling provenance:** measured by uPP → `tools/uarch_perf_probe/samples/gnr_pcl-gnrap01_demo.json (uPP: 6 SNC domains, 226 GB/s/domain, remote/local 0.61 — corroborates the SNC-corrected ceiling)`

**Model-level:** roofline target **60.6 TF/s** · measured **18.2 TF/s** · **30% of achievable**

Per-op, ranked by recoverable end-to-end fraction (shortfall × phase share):

| op | phase share % | regime | roofline (TF/s) | measured (TF/s) | efficiency | recoverable % |
|----|---------------|--------|-------------------|-------------------|------------|---------------|
| gate_up (MLP) | 30.0 | bw | 60.6 | 25.4 | 42% | 17.4 |
| down (MLP) | 30.0 | bw | 60.6 | 36.5 | 60% | 11.9 |
| qkv (attn) | 25.0 | bw | 60.6 | 35.4 | 58% | 10.4 |
| o (attn) | 15.0 | bw | 60.6 | 25.5 | 42% | 8.7 |

![roofline vs measured](./olmo2_7b_donor_roofline.png)

> Bars: roofline-achievable (target) vs measured per op. A tall gap on a high-share op is the top optimization RoI. `PENDING` = target published; measured fills in when the kernel runs.
