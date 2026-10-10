# Standalone Kernel Optimization Review

Reviewer: GitHub Copilot. Source baseline: `6080d61`; implementation reads were against that worktree. Existing measurements: job 384531, source `1e3096d77ca72eb3933c3c11b3003e8e9f325e09`, with native-source continuity established in the preceding reviews. A concurrent edit to F4 was present and was neither reviewed nor modified for this report.

## Scope and Verdict

This review concerns **standalone kernel algorithms, arithmetic, data movement, scratch storage, and their measured performance**. Dispatch/serial-parallel threshold tuning, engine integration, cross-operator fusion, model-level profiling, and end-to-end optimization are deferred as requested. They are not prerequisites for the experiments or stopping decisions below. Hold existing dispatch and thread configuration fixed when comparing standalone variants.

**The optimizations are worthwhile, but standalone optimization is not yet demonstrated complete for indexer logits, sparse attention, or compressor pooling. Top-k, Sinkhorn, and combine are reasonable provisional stopping points.** That is an allocation-of-effort judgment, not a claim that those three kernels are irreducible or fully correctness-certified.

The strongest remaining opportunities have specific mechanisms: remove work the indexer does not need; reduce compressor transcendental work; and use an existing BF16-input/FP32-output GEMM facility for source-faithful sparse attention. None requires an end-to-end run to evaluate. None has a measured speedup yet.

## Measured Baseline

The table reproduces the reviewed median across three process summaries from `/scratch/bkaul/replicated_all_384531.log`. Each process latency is a median of five 30-call means. Speedups use paired ratios, not ratios of these aggregated latencies. Conditions: EMR, 64-thread configuration, warm/reused inputs. Full reconstruction and comparator caveats are in [the round-3 review](phase1_kernel_implementation_review_round3.md).

| Kernel | M1 us | M8 us | M16 us | M32 us | M64 us | Standalone disposition |
|---|---:|---:|---:|---:|---:|---|
| Indexer logits | 102.1 | 202.5 | 219.1 | 268.2 | 427.7 | Keep open: representation, staging, epilogue work |
| Sparse best-of, existing paths | 73.7 | 167.2 | 256.3 | 392.1 | 729.1 | Good FP32 baseline; one important donor design untested |
| Compressor R128/D512 | 26.2 | 77.8 | 138.2 | 261.8 | 562.1 | Keep open: exponential work and missing R8 remeasurement |
| Indexer top-k | 2.6 | 11.9 | 12.1 | 12.9 | 13.6 | Provisionally sufficient; scratch is optional |
| Sinkhorn | 2.4 | 14.0 | 14.2 | 14.0 | 15.5 | Provisionally sufficient; no demonstrated large arithmetic gap |
| Combine | 2.3 | 11.5 | 11.5 | 11.8 | 13.6 | Provisionally sufficient; no demonstrated large remaining gap |

Current replicated timing is missing R8/D512, R8/D128, and direct BF16-KV indexer inputs. Historical job 384474 compressor vectors are R8/D512 = 16/20/25/28/42 us and R8/D128 = 13/14/14/16/20 us, ordered M1/8/16/32/64. These are historical observations, not substitutes for current replicated measurements.

### What Roofline Does and Does Not Establish

Nominal EMR anchors used in the existing analysis are 358.4 GB/s, BF16 AMX 124.5184 TF/s, and FP32 AVX512 7.7824 TF/s. They are hardware diagnostics, not op-specific measured achievable ceilings.

- Indexer M64: 427.7 us versus a 100.251 us useful-byte floor, about **4.27x**. This motivates conversion/packing/epilogue attribution; it does not prove a recoverable 4.27x speedup.
- R128 compressor M64: 562.1 us versus a 94.720 us byte-only floor, about **5.93x**. That floor omits exponential throughput and recurrence dependencies, so it cannot establish sixfold available gain.
- Sparse FP32 M32/M64: approximately **70.4%/75.7%** of nominal FP32 peak for the two leading GEMMs. This is a credible large-M FP32 implementation, but does not assess a different BF16 algorithm or the smaller-M regimes.
- Combine M64 implies about **386 GB/s** of useful boundary throughput, above nominal DRAM bandwidth. Reused cached data explains why a DRAM-only floor is not the right saturation claim.
- Top-k and Sinkhorn are governed by selection, tiny reductions/divisions, and fixed call costs. A tiny byte-only floor or a dense-GEMM peak is not their attainable ceiling.

A matched standalone donor/reference and component measurements at the actual working-set size are the appropriate empirical ceilings. These observations remain scoped to the recorded EMR configuration, not an unmeasured target machine.

## Opportunity 1: Indexer Data Movement and Epilogue

### Existing work worth retaining

[Indexer implementation](../../kernels/dsa_pilot/indexer_logits.cpp#L73) already delegates contraction to BF16-input/FP32-output library BRGEMM, packs Q once per request, converts FP32 KV tile-locally, and reduces scores without a whole-batch score tensor. Both current entry paths preserve the published BF16 score/product/final-logit boundaries. This is a sensible architecture; replacing the library GEMM with a new handwritten one is not the first justified step.

### I1: Remove unused BF16-KV staging allocation

**Evidence:** the parallel callback creates `Abuf(Sb * D)` unconditionally. When KV is BF16, the GEMM reads directly from `kvb`; `Abuf` is unused. The buffer is allocated/value-initialized per parallel callback, not once per GEMM tile.

**Why feasible:** make only that allocation conditional. The operands, GEMM, reduction order, and output remain unchanged. Removed work is concrete: allocation and initialization of `2 * Sb * D` bytes per callback. It applies to the tiled BF16-KV path; it does not imply a benefit to the separate M1 path or FP32-KV path.

**Why worthwhile:** this is a small, reversible experiment with low numerical risk, rather than a speculative algorithm rewrite. Expected magnitude is unknown and may be small. Keep it only if allocation/whole-call measurements show a repeatable benefit; allocator reuse or dominant GEMM time may make it immaterial.

### I2: Qualify the already implemented BF16-KV path

**Evidence:** current GEMM arithmetic already BF16-rounds KV, but the replicated sweep supplies FP32 KV. Direct BF16 KV avoids conversion and halves the dominant KV input read. At H64/D128/S1024, unique boundary bytes fall from `561408M` to `299264M`, approximately **46.7% less**. These figures exclude internal packing/scratch traffic.

**Why feasible:** the implementation and prior scoped numerical-equivalence checks already exist. The missing work is a paired standalone measurement using the same BF16-rounded values represented as FP32 versus BF16. Preserve FP32 output storage in both tests.

**Why worthwhile:** unlike a peak-based argument, this identifies bytes and conversion instructions the kernel actually avoids. It is still a conditional result: callers would have to supply BF16 KV to realize it. That integration decision is deferred. The FP32-input contract must retain its own performance result; do not present a storage-contract change as a free speedup for every caller.

### I3: Reduce redundant epilogue preparation only after attribution

**Evidence:** the inner score-row/head loops explicitly round the same per-head weights to BF16 for every score row, followed by BF16 score rounding, BF16 product rounding, and the head reduction. Q preparation also constructs a transposed contiguous tensor before packing.

**Feasible experiment:** first use the existing stage timing and focused epilogue/packing measurements, with instrumentation disabled for final timing. Inspect generated code: the compiler may already eliminate invariant work. If repeated weight conversion remains material, prepare BF16-rounded weights once per request and reuse them. Further packing-copy elimination needs evidence that its cost matters and that the library's layout contract permits it.

**Numerical constraint:** do not remove required score/product/logit rounding to gain speed. A SIMD reduction across heads can change addition order and induced top-k selection. Hoisting an invariant conversion is a much lower-risk experiment than reassociating the reduction. Recheck signed weights, near-cutoff selections, all M values, and tails.

**Verdict:** keep indexer open for these bounded tests. Do not change M dispatch or tune serial/parallel thresholds in this phase.

## Opportunity 2: Compressor Exponential Strategy

[The online update](../../kernels/dsa_pilot/compressor.cpp#L44) keeps channel-local max, denominator, and weighted accumulator, with no full probability tensor. N x D tiling and contiguous channel access are already good choices. The unresolved arithmetic cost is the recurrence's two `std::exp` expressions per ordinary update. A SIMD pragma does not prove those calls lower to efficient vector exponentials; inspect compiler output rather than assuming either scalar or vector code.

### C1: One exponential per finite update

For finite running maximum $m$ and new score $x$, let $m'=\max(m,x)$. The current factors are $c=\exp(m-m')$ and $w=\exp(x-m')$. One factor is exactly 1. Compute $t=\exp(-|x-m|)$ once, then choose:

$$
(c,w)=\begin{cases}(t,1),&x>m\\(1,t),&x\le m.\end{cases}
$$

**Why feasible:** this is an algebraic identity that retains the online recurrence, state footprint, loop order, and KV access pattern. Explicit first-valid and masked-lane handling must remain: initial $m=-\infty$, masked $x=-\infty$, and both masked cannot be sent through an unguarded infinity subtraction. Do not accidentally compute both exponential branches before selecting.

**Evidence gathered for this report:** one-exp and original two-exp factors agreed exactly for 100,000 FP32 PyTorch test pairs, including equal values, first-valid, masked, and both-masked cases. This is an algebra sanity check, not native C++ libm/vector-exp parity or a speed measurement.

**Why worthwhile:** it can approximately halve nontrivial exponential evaluations in this recurrence without another pass over input data. If exponentials occupy fraction $f$ of current time, an ideal halving of that component alone gives $1/(1-f/2)$, not an automatic 2x whole-kernel gain. Extra selects, compiler optimization of the current code, and non-exp costs can erase the benefit.

### C2: Use an established vector-exp implementation

**Why feasible:** channels are independent and contiguous, so SIMD can operate across D without reordering the reduction over R. The project already uses ATen's CPU vector facilities. A supported vector-exp implementation is preferable to authoring a new approximation polynomial.

**Why worthwhile:** batching independent exponentials can reduce scalar-call overhead and improve transcendental throughput. Compare generated code first; this is not an opportunity if the current compiler/library already produces equivalent vector math. Test C1 and vector-exp separately before combining them so each effect is attributable.

**Risk:** vector-exp need not match scalar libm bitwise. Keep FP32 state and published mask semantics, evaluate normal/heavy/extreme inputs and every R/D/M coordinate, and keep numerical screening distinct from acceptance. Do not enable broad fast-math flags that silently alter unrelated arithmetic.

### C3: Stable multipass pooling as the alternative algorithm

**Design:** first compute per-channel maximum over R. Then compute one exponential per element, accumulate denominator and weighted numerator, and divide once at the end. It need not materialize a full weights tensor. Compared with online normalization, it removes running-max rescaling of the accumulator but adds a score/APE pass.

**Why feasible:** the required state remains O(D); R8 and R128 are short fixed windows already supported by the kernel. At R8/D512, KV, score, and APE are each 16 KiB per corresponding window, giving a 48 KiB logical three-array footprint before state/output and sharing effects. That makes a cache-resident extra pass plausible, not guaranteed. Do not extrapolate that cache argument to R128 without measuring.

**Why it might win:** fewer exponentials than the original formulation and a simpler accumulation pass. **Why it might lose:** extra traffic, a second traversal, and different floating-point reduction/rescaling. Once C1 removes the redundant exp, multipass must earn its keep against that stronger baseline, not just the original two-exp implementation.

**Verdict:** compressor is the clearest small, algorithmically justified experiment. Start with C1, then evaluate vector-exp and multipass only as the measurements warrant. Replicate all three shapes at all five M values; do not let R128 results stand in for R8.

## Opportunity 3: Source-Faithful BF16 Sparse GEMMs

### Why the current FP32 result is not the end of the search

[The FP32 implementation](../../kernels/dsa_pilot/sparse_attend.cpp#L67) uses two library BMMs and an in-place fused softmax/sink pass. This is a strong reuse-first baseline. Its experimental BF16 sibling instead uses BF16 BMM outputs and normalized-weight rounding, unlike the published 64-block recurrence. Its loss does not test the proposed design below.

### S1: Two BF16-input/FP32-output GEMMs per source block

Preserve 64-entry block ordering, BF16 Q/KV operands, FP32 score accumulation and running state, BF16 rounding of each block's **unnormalized** exponentials before the value GEMM, sink placement after the block loop, and final BF16 output. The reference is [the existing blockwise adapter](../sparse_ref.py#L33), with its separate GPU-conformance limitations.

**Why feasible in this codebase:** indexer already calls `at::native::cpublas::brgemm` with BF16 A/B pointers and an FP32 C pointer, including the associated packing API. The required arithmetic facility therefore exists in the installed stack; this is not a proposal to invent a new GEMM engine. For H64/D512 and block64, the score GEMM is `(64 x 512) * (512 x 64)` and the value GEMM is `(64 x 64) * (64 x 512)`. Their dimensions are tile-friendly. Other TP-head counts still need separate evaluation.

**Why it could improve performance:** BF16 GEMM can exploit AMX while retaining FP32 accumulators; blockwise execution avoids a full `[M,H,K]` score allocation and its global softmax passes. At M64/H64/K512 the current score tensor is **8 MiB**; one H64/block64 FP32 score tile is **16 KiB**. Those are score-only sizes: per-task accumulators, Q/KV, packing, and concurrent workers still consume substantial memory. The output accumulator alone is 128 KiB for H64/D512, so do not claim the whole working set fits L1.

**What could defeat it:** at K512, eight source blocks require 16 small GEMM calls per request instead of two large ones. KV needs efficient packing in the orientations used by both GEMMs, and packed buffers must be amortized within the standalone call. Conversion, packing, BF16 probability casts, running-state rescaling, and call overhead may outweigh faster contraction. Nominal BF16/FP32 peak ratios do not predict whole-kernel speedup.

**Bounded experiment:** implement one direct standalone variant using the existing library, hold dispatch fixed, and account for all preparation inside measured latency. Benchmark M1/8/16/32/64 at K128/160/512/640 and the required head counts. Compare against both the existing native baseline and a reference with the same arithmetic boundaries. Include all-masked/padding behavior only under an explicit input contract; do not alter source block grouping by silently removing interspersed sentinels.

**Rejection criterion:** if a correctly qualified implementation loses after including packing/conversion, record that result and retain the FP32 baseline for the tested regime. Do not rescue the claim with GEMM-only timings or compare different numerical contracts without labelling them. No new dispatch policy is requested.

**Verdict:** highest-potential, highest-effort remaining standalone experiment. It is justified by an existing library capability and concrete intermediate traffic, not by a promise of reaching AMX peak.

## Lower-Priority Kernels

### Top-k

[The kernel](../../kernels/dsa_pilot/indexer_topk.cpp#L16) already uses selection rather than sorting all S entries and reuses row scratch within parallel callbacks. For S1024/k512, replacing this with a heap or a full sort has no demonstrated advantage. Persistent scratch reuse is a feasible optional allocation experiment, but cross-call storage adds concurrency/reentrancy concerns, and the core scan/partition work remains. The existing scratch is not allocated once per element or unconditionally once per row.

Provisionally stop mandatory standalone optimization here. Preserve selection/tie/mask checks; unsorted membership equivalence does not itself guarantee blockwise downstream rounding equivalence. Chunk/row dispatch tuning is explicitly deferred.

### Sinkhorn

[The fused recurrence](../../kernels/dsa_pilot/sinkhorn.cpp#L39) already keeps the hc4 matrix and normalization state local across all iterations. It removes the dominant framework-level sequence of small tensor operations. The next possible work would be fixed-hc specialization/register allocation or repeated-denominator transformations, but neither has demonstrated substantial standalone upside. Reciprocal-multiply replacement can change rounding; fewer iterations change the operation.

Provisionally stop mandatory standalone optimization. Preserve iteration count and epsilon placement; do not use a dense-GEMM peak to manufacture a large optimization gap. Thread/dispatch thresholds remain deferred.

### Combine

[Accumulate-once tiling](../../kernels/dsa_pilot/combine.cpp#L25) reads each input contribution once and avoids repeatedly updating the full output row. A register-vector formulation could potentially reduce L1 scratch traffic, but that is an optional instruction-level experiment, not a demonstrated large gap. It must retain per-element accumulation order and rounding, and may merely trade streaming efficiency for more live registers.

Provisionally stop mandatory standalone optimization. Do not describe reused-buffer throughput as proof of DRAM saturation. Cross-op fusion and dispatch changes remain deferred.

## Experiment Order and Stopping Rules

1. **Repair the standalone comparison, not the engine:** retain raw paired trials, match arithmetic/storage contracts, record the exact native build and library/thread identity, and explain comparator drift. Indexer M16 Torch variability and sparse M64's changed Torch time already show why ratios alone are insufficient. Keep cold/reused/rotating-buffer conditions explicitly separate.
2. **Run low-cost, directly grounded tests:** I1, I2, and C1. Attribute I3 before changing it. These do not require model execution or new dispatch heuristics.
3. **Pursue the larger alternatives conditionally:** C2/C3 and S1. A documented loss against a qualified stronger baseline is a valid completion result.
4. **Leave top-k, Sinkhorn, and combine provisionally closed for optimization effort.** Reopen only on evidence of an instruction/data-movement bottleneck or a concrete standalone candidate, not because a nominal roofline ratio looks large.

For each experiment, preserve the current native baseline, measure at all required standalone M/shape/dtype coordinates, include allocations/conversions/packing in whole-call latency, use at least three replicated process runs with raw paired observations, and report spread alongside medians. Keep arithmetic acceptance separate from speed. Indexer M8's observed 0.98x comparator median is unresolved; neither call it a tie nor hide it in an aggregate average. A failed candidate, no measurable benefit, or a measured tradeoff is a legitimate disposition; no experiment is required to produce a win.

**Completion criterion for this standalone phase:** each of the three open kernel families has a measured disposition for its justified opportunities, complete standalone measurement coverage, and honestly scoped numerical evidence. This is not a requirement to hit nominal peak and not a requirement to conduct end-to-end runs now. Performance closure does not waive the separate correctness/acceptance gates.

## Evidence Limits

This report includes source inspection of all six kernels and a cheap arithmetic sanity check, not new performance measurements. Existing timings and their three-process reconstruction were previously checked against the raw log. The 46.7% boundary-byte reduction, 8 MiB/16 KiB score sizes, and cited floor ratios were recomputed for this report. Proposed gains remain unmeasured; no expensive jobs, model runs, dispatch changes, native-code edits, policy changes, or modifications to the concurrent F4 work were made.