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


## 3. Kernel deep-dive: the compressor (softmax-pool)

A worked example of one authored kernel end-to-end. The compressor reduces a window of `R` raw KV tokens
(R = 128 or 8) to **one** compressed token via a **per-channel** softmax-weighted pool:

```
out[n,d] = Σ_r  softmax_r( score[n,r,d] + ape[r,d] ) · kv[n,r,d]
```

Each of the `D = 512` channels independently softmaxes over the window of `R` tokens and pools them.
Shapes: `kv, score = [N,R,D]`, `ape = [R,D]` → `out = [N,D]`, all fp32.

### 3.1 Roofline calculation (R=128, D=512, operating point M=32)

**Useful bytes moved** (each tensor streamed once, fp32 = 4 B):

| tensor | size | bytes |
|---|---|---:|
| kv `[M,R,D]` | 32·128·512 | 8.39 MB |
| score `[M,R,D]` | 32·128·512 | 8.39 MB |
| ape `[R,D]` (shared) | 128·512 | 0.26 MB |
| out `[M,D]` | 32·512 | 0.07 MB |
| **total** | | **17.10 MB** |

**FLOPs** ≈ `3·M·R·D` = 3·32·128·512 ≈ **6.3 MFLOP** (score+ape add, the exp, the weighted accumulate — a
*simplified* count that does not charge the exp its true transcendental cost; irrelevant here, see below).

**Arithmetic intensity** = 6.3M / 17.1M ≈ **0.37 FLOP/byte**, far below the ridge point
`peak/BW = 7.78 TF / 358.4 GB/s ≈ 21.7 FLOP/byte` → **firmly bandwidth-bound**.

**Floor = max(bytes/BW, FLOPs/peak):**
- BW term = 17.10 MB / 358.4 GB/s = **47.7 µs** ← binding
- compute term = 6.3 MFLOP / 7.78 TF = 0.81 µs

So **floor ≈ 47.7 µs** (nominal). Measured C/C++ = 176.1 µs → **27% of the nominal roofline**;
baseline torch = 408.7 µs → **C/C++ is 2.4× faster**.

### 3.2 Baseline implementation (the torch op it replaces)

[dsa_compressor_cpu.py](../plugin/intel_cpu_models/dsa_compressor_cpu.py):

```python
def compress_softmax_pool(kv, score, ape):
    w = (score.float() + ape.float().unsqueeze(0)).softmax(dim=1)   # materializes w[N,R,D]
    return (w * kv.float()).sum(dim=1)
```

Why it's slow — it is **memory-traffic-bound and makes many passes**:

- `score + ape` → one full `[N,R,D]` temporary (8 MB write).
- `.softmax(dim=1)` → internally max-over-R, subtract, **exp over the whole [N,R,D]**, sum-over-R, divide —
  several more full reads/writes of `[N,R,D]`, and it **materializes the weights tensor `w[N,R,D]` in DRAM**.
- `w * kv` → another `[N,R,D]` pass, then `.sum(dim=1)` → another.

Each is a separate ATen op with its own kernel launch; the `[N,R,D]` data is streamed through memory ~5–7×
plus the extra `w` allocation. On a BW-bound op that multiplies the ~17 MB useful traffic several-fold → ~409 µs.

### 3.3 C/C++ implementation + optimizations

[compressor.cpp](../plugin/kernels/dsa_pilot/compressor.cpp) — the inner loop:

```cpp
for (r = 0; r < R; ++r) {                 // ONE streaming pass over the window
  #pragma omp simd                        // (3) AVX-512 over the D channels
  for (j = 0; j < dd; ++j) {
    float x    = srow[j] + arow[j];
    bool  up   = x > m[j];
    float mnew = up ? x : m[j];
    float arg  = surv_masked ? -INF : (up ? (m[j]-x) : (x-m[j]));
    float t    = std::exp(arg);           // (4) C1: ONE exp (other factor is exp(0)=1)
    float corr = up ? t : 1.f, e = up ? 1.f : t;
    l[j]   = l[j]*corr + e;               // running denom
    acc[j] = acc[j]*corr + e*krow[j];     // running weighted accumulator
    m[j]   = mnew;                        // running max
  }
}
for (j) orow[j] = l[j] > 0 ? acc[j]/l[j] : 0;   // single divide at the end
```

Optimizations and why each helps:

1. **Streaming online ("flash") softmax — the big win.** The whole `softmax+pool` is fused into **one pass
   over R**, keeping only per-channel running `(max m, denom l, accumulator acc)`. The `w[N,R,D]` weights
   tensor is **never materialized**, and each input is **read once** instead of ~5–7 passes. On a BW-bound
   op this directly cuts the dominant traffic — it is why the kernel approaches its byte floor.
2. **N×D-tile parallelism.** Parallelizes over `N·ndt` tasks, not just `N`. The per-channel softmaxes are
   independent, so at small batch (M=1) the D dimension still fills all 64 cores — N-only parallelism left
   cores idle and actually *regressed* vs torch at M=1. Targets ~2×threads tasks.
3. **AVX-512 vectorization over D.** `#pragma omp simd` over the channel loop, tile width `DT` rounded to a
   16-lane AVX-512 register, so the running-state update is vector-wide.
4. **C1 — one `exp` per update.** In the online recurrence one of the two softmax rescale factors is always
   exactly `exp(0)=1`, so it is hardcoded and only the surviving `exp(min−max)` is computed. **Bit-exact** to
   the two-exp form (proven by building old/new and diffing), ~halves the transcendental work at R=128.
5. **Correctness details baked in.** Masked positions (`score = −∞`) contribute nothing (the `−∞` arg forces
   `exp→0` with no unguarded `inf−inf`); FP32 accumulation throughout; a single divide at the very end
   (all-masked channel → 0).

**Net:** the authoring win is almost entirely #1 (eliminate the DRAM weights temporary + collapse ~6 passes
into 1), amplified by #2–#3 so it is BW-limited rather than dispatch-limited; #4 is a small bit-exact
refinement on top. The remaining 73% gap to the nominal floor is grain / exp throughput (the datasheet BW is
a ceiling, not achievable), not another missing pass.


## Provenance
- Kernel latency + speedup: EMR node, **job 385001**, order-varied paired trials, 3-process medians; baseline = the torch op each kernel replaces at its arithmetic contract (refs noisy for indexer → medians).
- Roofline floor: analytic `max(bytes/BW, FLOPs/peak)` from each kernel's byte/FLOP contract vs the nominal datasheet peak.
- Correctness: F4 acceptance gate (PARTIAL, screening-only); GPU oracle (H200, job 384502) for the sparse reference dtype.

## Next: end-to-end (Phase-2)
Progressive end-to-end wall-time on the depth-reduced dummy-weight proxy (proxy4 + proxy8 → `t(L)=t_fixed+L·t_layer`
extrapolation to 43 layers), at the chosen operating point (throughput headline B=32 @ S=1024, plus the B=1
single-stream latency floor), integrating each kernel one at a time. The E2E tok/s curves will be added here.
