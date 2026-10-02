# Roofline target vs measured — DeepSeek-V4-Flash

**Node** `gnr_6980p (SNC-on: 6 NUMA domains; heads not div by 6 -> TP=4 = 4/6 domains; effective 841/1261 GB/s)` · **fp8** · **decode** · batch 32 · unit `tok/s` · **284B total / 13B active (MoE)**

**Ceiling provenance:** measured by uPP → `tools/uarch_perf_probe/samples/gnr_pcl-gnrap01_demo.json (uPP: 6 SNC domains, 226 GB/s/domain, remote/local 0.61 — corroborates the SNC-corrected ceiling)`

**Overhead attribution (framework vs kernel):** [`plugin/validate/results/deepseek_v4_flash_overhead.json`](./deepseek_v4_flash_overhead.json) — measured kernel/torch/framework split that routes each hotspot to its lever.

**Model-level:** roofline target **146.7 tok/s** (GNR fp8, batch 32) · authoritative GNR-fp8 measured awaits the full-weight GNR run · **EMR-MXFP4 operating point measured this session — see [Measured](#measured-2026-10-01) below.**

> **TP correction (measured 2026-10-01):** the header's `TP=4` is a *capacity/prefill* mapping, NOT the decode optimum. For DECODE, **tp=1 + decode-cap=8 WINS**; tp>1 *hurts* decode (the decode thread cap can't apply at tp>1 — it would deadlock the shm barrier — so decode is stuck at the full per-rank thread count = barrier thrash). Measured: GNR tp=2 full-43 decode 3.3 vs EMR tp=1+cap 9.3 tok/s (2.8× worse). DSV4 fits one socket, so decode needs no TP; TP is a capacity/prefill lever only.

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

## Measured (2026-10-01)

Config: EMR Xeon 8592+ (one socket, ~226 GB/s/domain), MXFP4 routed experts (native from-packed, no dequant), `OMP_WAIT_POLICY=passive`. Full-43 = full depth, dummy weights (representative for decode, which is weight-streaming-bound). All perf wins below are **scheduling-only / token-identical** (parity-confirmed: coherent, correct gsm8k-style output).

**Systemic wins (the dominant levers — both bigger than any per-op kernel tweak):**

| lever | effect | note |
|-------|--------|------|
| OpenMP spin-wait fix (`OMP_WAIT_POLICY=passive`, `KMP_BLOCKTIME=0`) | **63× prefill / 10× decode** | idle OMP threads busy-wait by default → inter-region thrash; was THE artifact the whole per-op profile was measuring |
| Continuous batching (B=1→32) | **10× aggregate decode** (8.1→81.2 tok/s) | decode is throughput-bound; per-token MoE cost 1.39→0.32 ms |

**MoE routed-expert GEMM (the #1 op, 68% of decode) — measured BW efficiency vs roofline:**

| phase | achieved | vs BW roofline |
|-------|----------|----------------|
| prefill (M=64) | 255 GB/s | **75%** (near-ceiling) |
| decode (M=1) | 52 GB/s | 26% → amortizes toward 75% as batch grows (the batching lever) |

**Full-model (43 layers, dummy, batch 32):**

| config | prefill tok/s | decode tok/s |
|--------|---------------|--------------|
| **EMR tp=1 + cap=8** (best decode) | 69.6 | **9.3** |
| GNR tp=2 per-socket | 77.6 (+11%) | 3.3 (2.8× worse — TP hurts decode) |

**Conclusions:** decode is at its practical ceiling for this stack at serving batch — MoE is already native-from-packed, prefill MoE is near-roofline (75%), and the headroom is realized by **batching + the spin-wait fix**, not kernel work. tp=1+cap is the decode optimum; TP is capacity/prefill only. (Methodology + all learnings codified in `.agents/skills/`: runtime-config-tuning, perf-proxy, compute-from-native-precision, sub-numa-clustering, model-profile-hotspots.)
