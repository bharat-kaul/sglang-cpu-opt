# Roofline target vs measured — DeepSeek-V4-Flash — routed-expert MoE kernel (68% of decode)

**Node** `EMR Xeon (1 NUMA domain; uPP stream_triad 226 GB/s)` · **MXFP4 W4A16 (fused fp4->bf16)** · **decode / prefill operating points** · batch 32 · unit `GB/s` · **284B total / 13B active (MoE); experts native MXFP4 (W4A16)**

**Ceiling provenance:** measured by uPP → `tools/uarch_perf_probe (uPP stream_triad: 226 GB/s per NUMA domain, EMR)`

**Model-level:** roofline target **226.0 GB/s** · measured **170.0 GB/s** · **75% of achievable**

Per-op, ranked by recoverable end-to-end fraction (shortfall × phase share):

| op | phase share % | regime | roofline (GB/s) | measured (GB/s) | efficiency | recoverable % |
|----|---------------|--------|-------------------|-------------------|------------|---------------|
| MoE experts — decode, UNBATCHED (M=1) | 68.0 | memory | 226.0 | 59.0 | 26% | 50.2 |
| MoE experts — prefill (M=64) | 68.0 | memory | 226.0 | 170.0 | 75% | 16.8 |
| MoE experts — decode, BATCHED (M=32) | 68.0 | memory | 226.0 | 170.0 | 75% | 16.8 |

![roofline vs measured](./deepseek_v4_flash_roofline.png)

> Bars: roofline-achievable (target) vs measured per op. A tall gap on a high-share op is the top optimization RoI. `PENDING` = target published; measured fills in when the kernel runs.

## Model-level measured (this session)

The chart above is the **dominant op** (routed-expert MoE = 68% of decode); here is the whole model.
All wins below are **scheduling-only / token-identical** (parity-confirmed — coherent, correct output).

**Full model, EMR, tp=1 + decode-cap=8, full 43 layers, batch 32:**

| phase | measured (tok/s) |
|-------|------------------|
| prefill | **69.6** |
| decode | **9.3** |

**Systemic levers that dominated (each bigger than any per-op kernel tweak — both now reusable skills):**

| lever | effect |
|-------|--------|
| OpenMP spin-wait fix (`OMP_WAIT_POLICY=passive`, `KMP_BLOCKTIME=0`) | **63× prefill / 10× decode** (the whole first per-op profile was a busy-wait artifact) |
| Continuous batching (B=1→32) | **10× aggregate decode** (amortizes weight streaming — the chart's M=1→M=32 recovery) |

**TP note (measured):** for decode, **tp=1 + decode-cap=8 is the optimum**; tp>1 *hurts* decode (the cap can't
apply at tp>1 → barrier thrash — GNR tp=2 decode 3.3 vs EMR tp=1 9.3 tok/s, 2.8× worse). TP is a
capacity/prefill lever only; DSV4-Flash fits one socket so decode needs none.

**Forward target (not yet run):** GNR fp8, batch-32 decode **146.7 tok/s** (higher BW + fp8) — the
authoritative GNR-fp8 measured awaits the full-weight GNR run.
