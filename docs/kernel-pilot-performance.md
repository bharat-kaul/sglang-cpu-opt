# DSv4-Flash DSA — C/C++ Kernel Pilot (Intel Xeon / EMR)

Standalone-kernel performance for the six authored CPU kernels of the DeepSeek-V4-Flash DSA path
(indexer logits, indexer top-k, compressor softmax-pool, sparse attention, MHC sinkhorn, MHC combine),
visualized **roofline vs baseline vs optimized**. All numbers are from the committed, self-identifying
artifacts [roofline_sweep.json](../plugin/validate/results/roofline_sweep.json) and
[standalone_opt.json](../plugin/validate/results/standalone_opt.json); regenerate the plots with
[make_kernel_pilot_plots.py](../plugin/validate/make_kernel_pilot_plots.py).

> **Read these honestly.** The roofline peak is the **nominal datasheet reference** (1.9 GHz base clock,
> BW 358.4 GB/s, AMX bf16 124.5 TF, FP32 7.78 TF) — **not a measured achievable ceiling**, so "% of
> roofline" is "% of nominal reference." The `regime` tag is the larger theoretical term (a diagnostic,
> not a measured bottleneck); cache-resident rows are not DRAM-saturated. These are **standalone
> microbench** results at each kernel's call contract — **not** end-to-end or a production certificate.
> Correctness is tracked separately and remains **PARTIAL** (F4 screening, thresholds UNRATIFIED,
> promotion **BLOCKED**).

## 1. How close the optimized kernels run to the nominal roofline

![Optimized kernels vs nominal-reference roofline](plots/achieved_vs_roofline.png)

Achieved % of the nominal-reference roofline across the M-sweep (EMR 8592+, OMP64, job 384677, 3-process
medians). Highlights at M64: **combine 97.5%** (BW, cache-resident — near its useful-byte ceiling),
**sparse/bestof 71.8%** (compute-bound fp32-bmm donor, genuinely near the FP32 reference), **compressor
R128/D512 ~26%**, **indexer logits 25.3%** (BW, convert/pack-bound at small M). `indexer_topk` and
`sinkhorn` are **selection/dispatch-bound** — a byte-or-FLOP roofline is the *wrong* ceiling for them, so
their low % is a diagnostic, not a recoverable gap.

## 2. Roofline floor vs baseline vs optimized (exemplar: compressor R128/D512)

![Compressor roofline vs baseline vs optimized](plots/compressor_3series.png)

The clean three-series view where all three align: **roofline floor** (nominal), **baseline** (pre-opt
kernel), **optimized** (C1 one-`exp` recurrence). C1 is **bit-exact** to the baseline (proven by building
old and new and diffing outputs) — a pure speed win of **1.29–1.62×** that scales with the exp count (R).

## 3. Standalone optimization: baseline → optimized speedup (kept wins)

![Standalone optimization speedups](plots/opt_speedup.png)

Order-varied paired trials, 3-process medians (jobs 384621/384675). Kept wins: **C1** compressor one-`exp`
(bit-exact; R128 1.29–1.62×, R8 up to 1.21×), **I1** indexer drop-unused-alloc (bit-exact; small but real
at large M), **I2** bf16-KV vs fp32-KV (1.09–1.51×, conditional on the caller supplying bf16 KV).

### Measured dispositions (full audit trail in [standalone_opt.json](../plugin/validate/results/standalone_opt.json))

| Experiment | Result | Disposition |
|---|---|---|
| I1 — drop unused bf16 staging alloc | bit-exact; M32/M64 ~1.03–1.04× | **Kept** |
| I2 — bf16-KV vs fp32-KV | 1.09–1.51× (caller-dependent) | **Conditional** |
| C1 — one `exp` per compressor update | bit-exact; R128 1.29–1.62× | **Kept** |
| I3 — hoist invariant weight rounding | ~1.00× (compiler already hoists) | **Neutral → reverted** |
| C2 — ATen vector-`exp` | 0.62–1.01× (slower) | **Rejected** |
| C3 — multipass pooling | ~1.0–1.14× (not bit-exact, marginal) | **Not promoted** |
| S1 — blockwise bf16 brgemm sparse | 0.08–0.65× (1.6–12.8× slower than the donor) | **Rejected (measured loss)** |

Only measured wins ship (I1 + C1, both bit-exact); C2/C3/S1 are retained as labelled, correctness-screened
experimental variants, out of the shipped dispatch. A measured loss/neutral is a valid disposition.

## Provenance
- Roofline sweep: EMR node `pcl-sprh11`, source `aaccef6`, 3-process medians (job 384677), `LEGACY_RECONSTRUCTED` aggregate.
- Optimization deltas: EMR, jobs 384621 (step-2 I1/I2/C1) + 384675 (step-3 I3/C2/C3) + 384676 (S1), order-varied paired, 3 process replicates.
- Correctness: F4 acceptance gate (PARTIAL, screening-only), GPU oracle (H200, job 384502) for the sparse reference dtype.

## Next: end-to-end (Phase-2)
Progressive end-to-end wall-time on the depth-reduced dummy-weight proxy (proxy4 + proxy8 → `t(L)=t_fixed+L·t_layer`
extrapolation to 43 layers), at the chosen operating point (throughput headline B=32 @ S=1024, plus the B=1
single-stream latency floor), integrating each kernel one at a time. Baseline calibration is in progress; the
E2E tok/s curves will be added here.
