# Time attribution (where the wall-clock goes) — DeepSeek-V4-Flash

**Node** `EMR 1-socket (tp=1, decode-cap=8)` · **MXFP4 W4A16 experts, bf16 compute** · batch 1 · unit `% of phase wall-time` — each phase sums to 100%.

## prefill

| op | bucket | time (s) | % of phase |
|----|--------|----------|------------|
| moe.expert_apply | kernel | 16.503 | 47.1 |
| dense.fp8_linear | kernel | 2.884 | 8.2 |
| dsa.mla.attend | torch | 2.678 | 7.6 |
| dsa.compressor | torch | 1.986 | 5.7 |
| dsa.indexer | torch | 1.541 | 4.4 |
| mhc.hc_pre | torch | 1.402 | 4.0 |
| mhc.hc_post | torch | 1.103 | 3.1 |
| mla.b.mask | framework | 0.803 | 2.3 |
| mla.b.qk | kernel | 0.638 | 1.8 |
| mla.b.softmax | kernel | 0.609 | 1.7 |
| mla.b.pv | kernel | 0.532 | 1.5 |
| mhc.combine | torch | 0.233 | 0.7 |
| mhc.sinkhorn | torch | 0.228 | 0.7 |
| mla.b.gather | framework | 0.060 | 0.2 |
| dense.bf16_linear | kernel | 0.028 | 0.1 |
| mla.b.cvt | framework | 0.003 | 0.0 |
| other / unattributed | other | 3.801 | 10.9 |

## decode

| op | bucket | time (s) | % of phase |
|----|--------|----------|------------|
| moe.expert_apply | kernel | 8.124 | 29.9 |
| dense.fp8_linear | kernel | 4.159 | 15.3 |
| dsa.mla.attend | torch | 2.100 | 7.7 |
| dsa.compressor | torch | 2.029 | 7.5 |
| mhc.hc_pre | torch | 1.807 | 6.6 |
| dsa.indexer | torch | 1.399 | 5.1 |
| mhc.sinkhorn | torch | 0.729 | 2.7 |
| mla.b.softmax | kernel | 0.540 | 2.0 |
| mla.b.pv | kernel | 0.503 | 1.8 |
| mla.b.qk | kernel | 0.476 | 1.7 |
| mla.b.gather | framework | 0.424 | 1.6 |
| mhc.hc_post | torch | 0.293 | 1.1 |
| dense.bf16_linear | kernel | 0.240 | 0.9 |
| mla.b.mask | framework | 0.124 | 0.5 |
| mhc.combine | torch | 0.089 | 0.3 |
| mla.b.cvt | framework | 0.020 | 0.1 |
| other / unattributed | other | 4.141 | 15.2 |

![time attribution](./deepseek_v4_flash_pivot.png)

> Each bar is one phase, segments = share of that phase's wall-time (sum = 100%). `other / unattributed` = phase wall − sum(wrapped ops); a big slice there means more ops need wrapping before trusting the split.
