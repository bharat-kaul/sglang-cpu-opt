# Phase-1 Kernel Implementation and Roofline Review

Independent reviewer: **GitHub Copilot**. Reviewed HEAD: `c67caec9bcb61de1abf9e2f8934042b89c2f02c1`; initially clean worktree. Three bounded, read-only reviews covered measurement reconstruction, five CPU kernels, and sparse GPU conformance. The parent checked controlling code and raw logs before consolidation.

## Verdict

**CHANGES REQUIRED for Phase-1 sign-off.** There are useful measured improvements, but the indexer still changes non-tie selections against the published BF16-stage computation, its new M=1 path bypasses guards, and F4 misses structural failures. Sparse GPU evidence is valuable but does not establish an intrinsic noise floor or ratify `4e-3`. Every-M no-regression and optimization exhaustion are not established.

This reviews [phase1_kernel_review.md](phase1_kernel_review.md), [../results/kernel_opt_queue.json](../results/kernel_opt_queue.json), the six C++ implementations, F4, and existing logs for **M=1/8/16/32/64**. It does not alter policy, the provenance ledger, author records, or the intentionally preserved baseline sweep. F4 remains PARTIAL; no numerical threshold is ratified here.

## Findings

### P1-F1: High - Indexer Oracle Hides Non-Tie Selection Errors

[../f4_acceptance.py](../f4_acceptance.py#L46) rounds inputs to BF16 but contracts, multiplies weights, and reduces in FP32. Published `Indexer.forward` at HF revision `60d8d70770c6776ff598c94bb586a859a38244f1` has BF16 output boundaries after the einsum, weighted product, and reduction. The live [../../kernels/dsa_pilot/indexer_logits.cpp](../../kernels/dsa_pilot/indexer_logits.cpp#L140) implements the diagnostic FP32-stage arithmetic, not those boundaries. The reference weights can be signed; nonnegative uniform F4 weights omit this important domain.

Executed seed-1 FP4-grid inputs, H64/D128/S1024, signed BF16-exact weights, against the published-stage replay: maximum logit error **0.09235763549804688** at every requested M; first-row top-k intersection **511/512**. The tie-aware checker counted **1/8/16/32/64 non-tie violations**, respectively. Independent signed BF16 random cases at S1031 produced **0/4/6/12/31** violations. Through actual in-memory `f4.run()`, the grid case returned **0/PARTIAL** with the current adapter and **2/FAIL** with the published-stage replay. This is not a demand for blanket bit-exactness or an arbitrary continuous tolerance: the declared hard selection gate fails when given the correct stage reference.

**Required:** correct and independently conform the oracle, preserve the published intermediate precision in the candidate, and requalify both dispatch paths and downstream selections. Current performance describes the current arithmetic only; it cannot approve a semantically different replacement. Published sources independently read: [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py), [kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py).

### P1-F2: High - M=1 Dispatch Bypasses Indexer Guards

[../../kernels/dsa_pilot/indexer_logits.cpp](../../kernels/dsa_pilot/indexer_logits.cpp#L207) dispatches directly into an unchecked specialization. Safe oversized-allocation probes accepted flattened weights, KV D129 with query D128, and mismatched KV batches; the tiled entry rejects all three. Undersized allocations can reach unchecked raw-pointer reads; dangerous out-of-bounds calls were not executed. Validate the common contract before dispatch. Separately, N=0 reaches division by N in [indexer tiling](../../kernels/dsa_pilot/indexer_logits.cpp#L95) and [compressor tiling](../../kernels/dsa_pilot/compressor.cpp#L31); define empty-batch behavior or reject it explicitly. Those divisions were identified statically.

### P1-F3: High - F4 Structural Checks Can Be Bypassed

The [tuple evaluator](../f4_acceptance.py#L368) uses truncating `zip` without checking tuple length, per-output shapes, or dtypes. Full in-memory `run()` returned **0/PARTIAL** for an empty Sinkhorn result, a missing combination matrix, FP64 outputs, and an extra leading output dimension. An empty manifest also returns zero. These are mandatory structural/coverage checks, independent of continuous tolerance ratification.

All-masked compressor input returns C++ zeros while the published-stage reference produces NaNs; [continuous evaluation](../f4_acceptance.py#L376) records `max_err=nan` without a hard failure. Either establish that this input is outside the captured caller domain and reject it, or define its contract explicitly. Nonfinite reference/comparison evidence must not count as successful conformance. Require output-structure checks before repeatability/metrics and an explicit nonempty expected case inventory.

### P1-F4: High - Sparse Replica Error Is Not Intrinsic GPU Noise

The [BF16 replica](../f4_acceptance.py#L78) globally normalizes, including the sink in the maximum. The published GPU kernel uses **64-entry blockwise** online normalization, BF16-rounds each block's unnormalized exponentials, rescales FP32 accumulators, and adds the sink afterward. The replica is not a source-faithful recurrence.

Additionally, [GPU execution](../test_sparse_gpu_oracle.py#L68) receives BF16-rounded operands while [CPU Part B](../test_sparse_cpu_vs_gpu.py#L31) receives original FP32 operands. At M64, matching input rounding reduces bestof maximum error from **0.003457636 to 0.001458526**. A source-shaped blockwise CPU replica improves M64 double-precision cosine from **0.999997983314 to 0.999999979351**, though it is still not bit-exact. Neither discrepancy measures hardware nondeterminism; repeated GPU executions were not used to characterize noise.

[SPARSE_NOISE_FLOOR](../f4_acceptance.py#L96) is observed candidate error plus margin, whereas [../results/acceptance_policy.json](../results/acceptance_policy.json#L3) remains UNRATIFIED. The [sparse evaluator](../f4_acceptance.py#L354) annotates acceptance using max-absolute error alone and never makes an over-floor continuous result a hard failure. A deterministic output filled with 1000 passed through actual `run()` with `max_err=1.000e+03`, return 0/PARTIAL. Nongating continuous diagnostics are consistent with PARTIAL, **not** evidence for "accepted correctness." The [budget summary](../f4_acceptance.py#L478) also lists a path as accepted when any record accepts, rather than requiring all records.

**Required:** compare identical operands and the same output boundary against the source-faithful primitive; report approximation error separately from repeatability. Keep `4e-3` a proposal, not a ratified noise floor. Any approximate class requires an independently justified downstream budget and validation against it.

### P1-F5: High - Correctness Coverage Does Not Match the Sweep

The [F4 manifest](../f4_acceptance.py#L173) covers indexer/top-k at M1/8/32, compressor at M8, Sinkhorn/combine at M1/8, and sparse at M8/K512. Benchmark cosine at other M values is not the missing acceptance gate. Direct BF16-KV, signed indexer weights, noncontiguous layouts, and caller sentinel transformations need explicit coverage.

Actual [GPU cases](../test_sparse_gpu_oracle.py#L87) cover M1/8/64, H64/D512/K512, batch=1 with sequence length M, shared KV and identity indices. They do not certify independent request batches, K128/640/160 window-plus-compressed unions, causal sentinels, offsets, reordered gathers, or TP-local heads. The queue itself names those K values as prerequisites. FP32 output from a pre-gathered fragment is not the published BF16 output boundary or complete sparse path. Capture real call tensors before using these synthetic contracts for deployment ROI; a green shape-generic test does not prove shape provenance.

### P1-F6: High - Universal No-Regression Is Not Established

The claimed hard >=1.0x Torch floor conflicts with indexer M1: job384474 logs **96 us Torch / 100 us C++ = 0.96x**, and job384482 logs **95/102 us = 0.94x**. Job384476 records parity. "Within 33% noise" is not a paired equivalence or noninferiority test; these data establish neither a statistically significant regression nor a proven no-regression floor.

[Indexer timing](../bench_idx_logits.py#L24) averages 30 sequential calls, sparse averages 20, and other benchmarks average 50 after one warmup. Optimized launchers run each benchmark once, without trial distributions or randomized pairing; a printed "median" label does not change the estimator. All four principal logs begin with failed git-root lookup and record no source/build digest. Repeated jobs corroborate behavior but are not verified same-source replicated medians. The older three-repetition baseline remains valid as a historical baseline, not as replication of these optimized paths.

The author's indexer headline mixes M1 from job384474, other speedups from384482, and off-roof values from earlier384469. At384482, FP32-I/O off-roof factors are **65.12/16.04/8.34/4.67/3.76x** using the stated useful-byte ideal. Preserve individual measurement generations and their contracts.

### P1-F7: High - GPU Wrapper Can Report Success on Failure

[../run_sparse_gpu_oracle.sbatch](../run_sparse_gpu_oracle.sbatch#L29) pipes the container through `tail` without `pipefail` and then echoes the pipeline status. Job384501 recorded `Error: creating events dirs: mkdir /run/user/10788806: permission denied` followed by `[exit=0]`. Job384502 **did** execute the real primitive successfully; the finding does not invalidate that observation.

The successful run imports a mutable snapshot and writes a fixed overwriteable tensor file without source hashes, run identity, or a container digest. Current snapshot sources match the pin; that does not bind historical execution to the reviewed source. The script asserts no numerical acceptance. Propagate failures and bind source, binary/container, inputs, outputs, and job identity before treating this as a reusable conformance gate.

### P1-F8: Medium - Exactness, Saturation, and Exhaustion Are Overstated

Cosine-only [sparse timing checks](../bench_sparse_attend.py#L47) cannot establish "bit-identical" behavior. At M1, scalar differs from FP32 reference by **4.47034836e-7**, and scalar versus bmm by **3.87430191e-7**. Repeating identical data at N2 switches bestof to bmm and changes the output by the latter amount; same-path repetition is equal. Logged `cos_bo=1.000061` at M64 illustrates floating-point metric error, not exactness. In contrast, the direct-BF16-KV experiment does use `torch.equal`, for its sampled inputs only.

Sparse's 9.08x M8 speedup includes comparator instability: Torch is **1.336 ms** in job384476 versus **0.320 ms** in job384474. Optimized bestof is 0.147 ms in job384476. Resolve the comparator difference before assigning all of the ratio to fusion. Whole-op nominal FP32 utilization is approximately **13/47/56/66/74%**, not one uniform 64% GEMM-efficiency result.

Combine M64 useful traffic divided by 14 us is approximately **375 GB/s**, above the nominal 358.4 GB/s DRAM anchor. Repeated warm buffers can be cache-resident; this is not proof of DRAM saturation. R128 compressor M64 is approximately 61 GB/s, 17% of nominal, not uniformly 7%. Its current FP32-expanding implementation does not exclude a native BF16-input donor, although published FP32 state semantics constrain whether such a donor would be admissible.

Rejecting the existing sparse AMX variant, which rounds score output and normalized weights, does not reject tiled library BF16-input/FP32-output GEMMs with the published recurrence. That interface already exists in the indexer. Full score materialization is not proved irreducible; custom hand-written flash is not proved the only remaining path. Also, [sparse scale selection](../../kernels/dsa_pilot/sparse_attend.cpp#L23) silently maps NaN to default scale; a local probe confirmed identical output. Reject nonfinite scalars while preserving any intentional default sentinel.

## Reconstructed M Sweep

All vectors are ordered **M=[1,8,16,32,64]**, all latencies in **microseconds**. Rounded raw times and separately printed speedups may not divide exactly. These are selected, explicitly identified observations, not pooled medians or source-bound results.

| Kernel / record | C++ latency | Torch latency | Recorded Torch speedup |
|---|---|---|---|
| Indexer FP32-I/O / D | 102,201,209,234,377 | 95,561,811,852,967 | 0.94,2.79,3.88,3.64,2.56 |
| Indexer BF16-KV / D | 69,188,196,206,230 | Not timed for this contract | Not established |
| Sparse bestof / B | 66,147,247,416,749 | 239,1336,393,557,804 | 3.62,9.08,1.59,1.34,1.07 |
| Compressor R128/D512 / A | 25,79,141,271,555 | 220,2245,1655,1328,1433 | 8.64,28.51,11.77,4.91,2.58 |
| Compressor R8/D512 / A | 16,20,25,28,42 | 41,99,126,125,134 | 2.57,4.82,5.09,4.50,3.22 |
| Compressor R8/D128 / A | 13,14,14,16,20 | 26,31,38,75,109 | 1.92,2.22,2.62,4.61,5.50 |
| Top-k / C | 2,12,13,13,14 | 5,51,147,294,407 | 2.29,4.24,11.25,22.19,28.13 |
| Sinkhorn / C | 2,13,13,14,15 | 228,263,276,305,349 | 98.30,19.63,20.64,21.52,22.99 |
| Combine / C | 2,12,11,12,14 | 12,26,26,26,27 | 5.52,2.25,2.31,2.15,1.98 |

BF16-KV's comparator is FP32-KV **C++**, not Torch: D reports 98/193/210/239/348 us and printed ratios 1.43/1.03/1.09/1.13/1.47. Ratios use additional timed loops; BF16 conversion is outside timing. Caller-level savings depend on whether BF16 KV is already available under the correct contract. Sinkhorn/combine were already approximately this fast in the historical C++ baseline; their large Torch ratios must not be credited as newly demonstrated Phase-1 gains.

Raw records, external to the repository:

```text
A /scratch/bkaul/idx_breakdown_384474.log  compressor lines17-37; indexer41-45; sparse50-54
B /scratch/bkaul/idx_breakdown_384476.log  indexer41-45; sparse50-54
C /scratch/bkaul/rem_kernels_384480.log    top-k6-10; Sinkhorn14-18; combine22-26
D /scratch/bkaul/idx_breakdown_384482.log  indexer41-45; sparse50-54; BF16-KV57-61
E /scratch/bkaul/idx_breakdown_384469.log  earlier-pass indexer43-47
F /scratch/bkaul/sparse_cpu_vs_gpu_384505.log  CPU/GPU comparisons3-16
```

A-D report `pcl-sprh01.sc.intel.com`; E reports `pcl-sprh03`. Current launchers request EMR/DDR5600, an exclusive node, 64 threads, close/core OpenMP binding and NUMA0 CPU/memory binding. Builds use `-O3 -fopenmp -march=native`. Logs do not record actual thread count, affinity, loaded-library identity, or source digest. Seeded warm buffers are reused without cache eviction. These are EMR observations, not final GNR measurements.

### Nominal Roofline Interpretation

Use 358.4 GB/s, 124.5184 TF/s BF16 AMX, 7.7824 TF/s FP32 AVX512. Ideal is `max(useful_bytes/BW, useful_FLOPs/precision_peak)`, not a guaranteed achievable time. Sparse is dominated by its two FP32 GEMMs; the other displayed bounds are useful-byte times. Allocation, packing, dispatch, executed intermediates, cache traffic, comparisons, division and exponential throughput can dominate in practice. In particular, top-k and Sinkhorn cannot be judged against dense FMA peak.

| Contract | Useful bytes/call | Nominal ideal us at M1/8/16/32/64 |
|---|---|---|
| Indexer H64/D128/S1024 FP32-I/O | 561408M | 1.566,12.531,25.063,50.126,100.251 |
| Indexer same, BF16-KV only | 299264M | 0.835,6.680,13.360,26.720,53.440 |
| Sparse H64/K512/D512 FP32 | 1310720M+256 | 8.623,68.985,137.971,275.941,551.882 |
| Compressor R128/D512 FP32 | 4[(2RD+D)M+RD] | 2.200,12.480,24.229,47.726,94.720 |
| Compressor R8/D512 FP32 | Same formula | 0.143,0.823,1.600,3.154,6.263 |
| Compressor R8/D128 FP32 | Same formula | 0.036,0.206,0.400,0.789,1.566 |
| Top-k S1024/k512, int64 output | 8192M | 0.023,0.183,0.366,0.731,1.463 |
| Sinkhorn hc4/20 iterations | 192M+108 | 0.000837,0.004587,0.008873,0.017444,0.034587 |
| Combine hc4/H4096 FP32 | 81936M | 0.229,1.829,3.658,7.316,14.631 |

Shared operands are counted once. Indexer has 16777216M BF16 GEMM FLOPs plus 130048M FP32 reduction operations and ReLU comparisons; sparse has 67108864M leading FP32 GEMM FLOPs. Compressor's mathematical pool has (5R-1)DM ordinary operations plus RDM exponentials/comparisons, whereas the online implementation evaluates two exponentials per element. Combine has 28672M FLOPs. Do not price FP32 epilogues at AMX peak or mistake useful bytes for observed DRAM traffic.

## Independent Correctness Evidence

Single-thread synthetic probes used `/scratch/bkaul/venvs/sglang-cpu/bin/python`, `OMP_NUM_THREADS=1`, `MAX_JOBS=2`, and the specified C++ flags. These are cheap correctness probes, not new performance measurements, captured-shape evidence, or full-model gates.

| Kernel | Observed result across requested M | Limit |
|---|---|---|
| Indexer | Published-stage failures in P1-F1; sampled BF16-KV equivalence and replicated-grid M1/tiled equality passed | Does not conform to published selection contract |
| Compressor | 60 normal/heavy/masked cases, all three shapes plus D131 tail; max error7.63e-6; repeatable | All-masked contract unresolved; no ratified budget |
| Top-k | 75 random/tied/masked cases including k0/S/>S and empty S; zero membership violations; caller sentinel checks passed | Single-thread probes do not exercise chunked parallel dispatch |
| Sinkhorn | Published recurrence/epsilon placement matched; normal/heavy max error2.38e-7 | GPU bitwise conformance not established |
| Combine | H4096/1031, normal/heavy/strided; normal max error5.96e-7 vs multiply-then-sum | Not bit-exact to published FP32 expression |
| Sparse | Existing saved GPU outputs replayed; strided K128/160/640 at N1/2 passed FP32 mathematical checks | Not complete GPU gather/mask or independent-batch conformance |

Other exercised small-kernel guards and noncontiguous equivalence checks passed. No obsolete fixed guard defects are reopened. Multithread packing, chunked selection and race safety remain outside the single-thread probe coverage.

Saved GPU-output comparison, maximum absolute error:

| M | Bestof original FP32 inputs | Bestof matched BF16-rounded inputs |
|---|---:|---:|
| 1 | 0.002085537 | 0.001340300 |
| 8 | 0.002244294 | 0.001284570 |
| 16 | 0.002282351 | 0.001221061 |
| 32 | 0.003457636 | 0.001229644 |
| 64 | 0.003457636 | 0.001458526 |

M16/32 are slices of saved M64, **not independent GPU cases**. Tensors were loaded with `weights_only=True`. Raw job384505 reports every `<=floor?` comparison as `no` against 1.953e-3, including M64 bestof 3.458e-3 and AMX 5.859e-3. The field named `mae` is maximum, not mean, absolute error. Raising the comparison bound to 4e-3 is a policy choice, not a new noise measurement.

## Remaining Opportunities

Ranked by plausible deployment contribution and the cheapest discriminating experiment, not by Torch speedup alone. Numerical and measurement repairs above precede performance acceptance; experiments can proceed without claiming they are accepted replacements.

1. **Indexer precision-qualified staging and M1 dispatch.** Correct the stage contract first. Make BF16-KV feasibility a caller-level question; conditionally avoid the unused [Abuf allocation](../../kernels/dsa_pilot/indexer_logits.cpp#L116), then test reusable scratch/packing. Attribute conversion, allocation, packing, GEMM and epilogue separately: the current "compute" timer includes conversion and epilogue. M1 [materializes full scores](../../kernels/dsa_pilot/indexer_logits.cpp#L184), contrary to blanket L1-only fusion claims. Compare whole-S versus tiled execution over captured contexts with identical precision. At M64 and21 calls, current standalone accounting is7.92 ms FP32-I/O or4.83 ms BF16-KV; the difference is conditional, not measured E2E savings.
2. **Sparse source-faithful tiled library donor.** Resolve comparator instability and cover K128/640/160 first. Test BF16-input/FP32-output library GEMMs with the published blockwise recurrence and exponential rounding; compare against both saved GPU outputs and current donor latency. Reuse packing/cache-resident tiles where valid. Do not flatten independent KV sets into a shared pool. Existing AMX rejection does not eliminate this experiment, but neither a gain nor a16x speedup is promised.
3. **MHC parallel grain and caller fusion.** Sinkhorn/combine jump from about2 us at M1 to12-15 us at larger M. Compare serial versus parallel grain and then caller-level fusion/conversion with the same fixed thread pool. At86 calls each, M64 standalone accounting is1.29 ms Sinkhorn plus1.20 ms combine. Measure actual caller overhead before authoring a larger fused kernel; warm-cache useful bandwidth does not establish exhaustion.
4. **Compressor R8 vector-exp/grain, then R128.** [The inner loop](../../kernels/dsa_pilot/compressor.cpp#L51) has two exponentials per element and a serial recurrence over R. Compare a vector-exp donor or stable multipass formulation, preserving FP32 state and masks, against extra traffic and rounding effects. Amortized M64 pool costs are approximately0.087 ms R128,0.221 ms R8/D512,0.105 ms R8/D128 using the existing ratio-weighted call model. The largest individual R128 call is not automatically the largest deployment opportunity.
5. **Top-k scratch only after larger contributors.** Evaluate reusable [selection scratch](../../kernels/dsa_pilot/indexer_topk.cpp#L33) and row/chunk dispatch at captured S and k. Require exact non-tie membership and proper tie handling. Current M64 accounting is about0.294 ms at21 calls, limiting plausible total savings.

## Exit Criteria

1. Repair the source-stage indexer oracle/candidate and common guards; make F4 reject malformed structures, nonfinite comparison evidence, empty/incomplete inventories, and hard selection failures.
2. Extend acceptance to every required M and relevant captured shapes/layouts, with signed weights and real gather/mask semantics. Conform the sparse reference on identical operands/output boundaries; keep approximate budgets explicitly pending until independently justified and ratified.
3. Bind each run to source/build/library/config/input identity. Reproduce all M with at least three independent trials, report medians and variability, randomize or alternate paired comparisons, and separate cold-start from steady-state. Investigate M1 indexer and M8 sparse comparator behavior before claiming a universal floor. Use fixed affinity/thread pools and the appropriate target node; do not launch final target-HW performance on portable correctness infrastructure.
4. Preserve the historical baseline and publish optimized measurements as a separate versioned record. Price remaining opportunities by the actual caller workload, not microbenchmark speedup; proceed to captured integration/full-model gates without mistaking these proxies for certification.

No new cluster/GPU jobs, model loads, or performance runs were launched for this review. Only this reviewer report is authored; no production fixes, ledger changes, or tolerance ratification are included. The scope is sufficiently resolved to request the changes above, not to continue arbitrary malformed-input exploration.