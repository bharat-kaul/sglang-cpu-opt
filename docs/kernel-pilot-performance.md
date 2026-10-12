# DSv4-Flash DSA — C/C++ Kernel Pilot (Intel Xeon / EMR)

Performance for the six authored CPU kernels of the DeepSeek-V4-Flash DSA path at the serving operating
point **(M=32 decode batch, input S=1024, output G=256)**: for each kernel we show the **roofline floor**,
the **baseline** (the torch op it replaces), and the **C/C++ speedup** — with oneDNN/MKL use called out.
Numbers from the committed [cpp_vs_torch.json](../plugin/validate/results/cpp_vs_torch.json) (EMR job 385001,
3-process medians via [bench_replicated.py](../plugin/validate/bench_replicated.py)); regenerate with
[make_kernel_pilot_plots.py](../plugin/validate/make_kernel_pilot_plots.py).

> **Caveats.** The roofline peak is the **nominal datasheet reference** (BW 358.4 GB/s, AMX bf16 124.5 TF,
> FP32 7.78 TF @ 1.9 GHz base) — **not a measured achievable ceiling**, so "% of roofline" is "% of nominal
> reference." `indexer logits` and `sparse attend` (marked **†**) reuse a **oneDNN/MKL GEMM donor** and
> author only the fused epilogue — their speedup includes the library matmul; the other four are
> **pure-authored** (no library GEMM). These are **standalone microbench** results (indexer/top-k at the
> real S=1024 KV length; compressor/sparse/sinkhorn/combine on fixed intermediate sizes). Correctness is
> tracked separately and remains **PARTIAL** (promotion **BLOCKED**).

## 1. Per-kernel at the operating point (M=32, S=1024): roofline vs baseline vs C/C++

![Per-kernel roofline vs baseline vs C/C++ at M=32](plots/operating_point.png)

| kernel | implementation | baseline (torch) µs | C/C++ µs | **speedup** | roofline floor µs | % of roofline |
|---|---|---:|---:|---:|---:|---:|
| indexer logits † | oneDNN BRGEMM + authored epilogue | 558.3 | 272.4 | **2.1×** | 50.1 | 18% |
| indexer top-k | pure-authored (selection) | 325.8 | 13.2 | **24.4×** | 0.73 | 6% |
| compressor | pure-authored (AVX-512 online-softmax) | 408.7 | 176.1 | **2.4×** | 47.7 | 27% |
| sparse attend † | oneDNN `at::bmm` + authored softmax/sink | 542.1 | 402.7 | **1.4×** | 275.9 | 69% |
| MHC sinkhorn | pure-authored (fused recurrence) | 316.4 | 13.6 | **21.5×** | 0.017 | 0% |
| MHC combine | pure-authored (accumulate-once) | 27.2 | 11.7 | **2.3×** | 7.3 | 63% |

The two **oneDNN-donor** kernels (†) are GEMM-heavy; the C/C++ gain there is the fused epilogue over the
library matmul. The four **pure-authored** kernels get the large gains (sinkhorn 21.5×, top-k 24.4× — torch
pays ~40 tiny ops / a full sort that the fused kernel collapses). `sparse attend` is **compute-bound** (69%
of the FP32 roof) and `combine` is near its BW ceiling (63%, cache-resident), so their speedups are smaller
by design — they were already close to the library/hardware limit.

## 2. % of nominal roofline across the M-sweep (1→64): arithmetic intensity

![% of nominal roofline vs M](plots/roofline_msweep.png)

As the decode batch M grows, arithmetic intensity rises and the GEMM/BW-bound kernels climb toward their
roofline: **combine → 100%**, **sparse → 74%**, **compressor** plateaus ~27% (grain/exp-bound), **indexer
logits** 25%. `indexer top-k` (~11%) and `sinkhorn` (~0%) stay low because they are **selection/dispatch-
bound** — a byte-or-FLOP roofline is the wrong ceiling for them, so the low % is a diagnostic, not a
recoverable gap. This is the "how close to the metal, and does more work help?" view.


## Provenance
- Kernel latency + speedup: EMR node, **job 385001**, order-varied paired trials, 3-process medians; baseline = the torch op each kernel replaces at its arithmetic contract (refs noisy for indexer → medians).
- Roofline floor: analytic `max(bytes/BW, FLOPs/peak)` from each kernel's byte/FLOP contract vs the nominal datasheet peak.
- Correctness: F4 acceptance gate (PARTIAL, screening-only); GPU oracle (H200, job 384502) for the sparse reference dtype.

## Next: end-to-end (Phase-2)
Progressive end-to-end wall-time on the depth-reduced dummy-weight proxy (proxy4 + proxy8 → `t(L)=t_fixed+L·t_layer`
extrapolation to 43 layers), at the chosen operating point (throughput headline B=32 @ S=1024, plus the B=1
single-stream latency floor), integrating each kernel one at a time. The E2E tok/s curves will be added here.
