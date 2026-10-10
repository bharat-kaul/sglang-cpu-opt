# Response to the Standalone Kernel Optimization Review (`b51d1eb`)

Addressing [standalone_kernel_optimization_review.md](plugin/validate/reports/standalone_kernel_optimization_review.md) (`b51d1eb`). All seven named experiments (I1, I2, I3, C1, C2, C3, S1) are now implemented and measured per the review's experiment order and stopping rules. Measured dispositions are recorded in [standalone_opt.json](plugin/validate/results/standalone_opt.json). Commits: `595c609` (I1/C1 + rig), `3ed5869` (step-2 dispositions), `643834f` (I3/C2/C3), `a462c2a` (S1), `0e8c186` (step-3 dispositions + I3 revert).

**Method (review step 1, "repair the comparison, not the engine"):** every result is an order-varied **paired** old-vs-new trial (new/old alternated each trial), **cross-process median of 3 process replicates** on EMR (`--constraint=ddr5600 --exclusive`, `OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores`, `numactl -N0 -m0`), reporting spread alongside medians, with run identity (HEAD, dirty, threads, baseline ref) stamped in each log. Correctness is held **separate** from speed: bit-exact changes are proven by building old and new and diffing outputs; non-bit-exact variants are F4-screened. Baselines are read from pinned git refs so the native build identity is explicit.

## Measured dispositions

| Exp | Change | Correctness | Speed (cross-process median) | Disposition |
|---|---|---|---|---|
| **I1** | drop unused BF16 staging alloc on the bf16-KV path | **bit-exact** vs `020f429` | M32 1.04x, M64 1.03x; ~1.0 small M | **Kept** (small, real large-M; free) |
| **I2** | qualify existing bf16-KV path (same values FP32 vs BF16) | representation-equivalent | **1.09–1.51x** (M1 1.44, M64 1.51) | **Conditional** (caller supplies bf16 KV; integration deferred) |
| **C1** | one `std::exp` per compressor update | **bit-exact** vs `020f429` | R128 1.29–1.62x; R8 1.0–1.21x | **Kept** (scales with the exp count) |
| **I3** | hoist invariant per-head BF16 weight rounding | bit-exact vs HEAD | **~1.00x all M** (bf16kv + fp32kv) | **Neutral → reverted** (compiler already hoists) |
| **C2** | ATen vector-exp (one-exp, vectorized) | screened ≤7e-7 vs oracle | **0.62–1.01x — slower everywhere** | **Rejected** (`Vectorized::exp` loses to scalar) |
| **C3** | stable two-pass pooling (max, one exp/elem, divide) | screened ≤7e-7 vs oracle | R128 large-M ~1.08–1.14x; neutral/loss at R8 | **Not promoted** (doesn't decisively beat bit-exact C1; not bit-exact) |
| **S1** | source-faithful 64-block **bf16-in/fp32-out brgemm** flash | screened ≤3.9e-3 vs the blockwise ref (cos ~0.9999) | **0.08–0.65x = 1.6×–12.8× slower** than the fp32-bmm donor at every (M,K) | **Rejected** (measured loss, the review's anticipated outcome) |

The shipped kernels therefore carry **only the measured wins**: I1 (bit-exact alloc drop) and C1 (bit-exact exp halving). I2 is a property of the already-shipped bf16-KV path (not a code change). C2/C3/S1 are retained as **clearly labelled experimental, correctness-screened** variants ([compressor_softmax_pool_vexp](plugin/kernels/dsa_pilot/compressor.cpp#L80), [compressor_softmax_pool_multipass](plugin/kernels/dsa_pilot/compressor.cpp#L152), [sparse_attend_blockbf16](plugin/kernels/dsa_pilot/sparse_attend.cpp#L143)) that are **not wired into the shipped dispatch** — the same treatment as the dominated `amx` path — so a reviewer can reproduce each rejection.

## Notes on the two "conditional, larger" findings

- **S1 (Opportunity 3):** implemented faithfully against [the blockwise adapter](plugin/validate/sparse_ref.py#L33) — bf16 operands, fp32 accumulate in both GEMMs, bf16 cast of the **unnormalized** per-block exponentials before the value GEMM, per-block fp32 rescale, sink after the loop with the final running max, bf16 output. A concrete structural finding surfaced: the cpublas **pack ukernel caps N at 64** (the AMX tile width), so the value GEMM (N=D=512) cannot pack directly; I reframed it as the **transposed** accumulator `accT[D,H] = kvblkᵀ[D,blk] @ wblkᵀ[blk,H]` (N=H=64), reusing the transposed KV tile as both the score-B and the value-A. Even so, the per-block transpose + two packs + two micro-GEMMs (× K/64 blocks) + fp32 rescale decisively lose to the two large MKL bmms of the fp32-bmm donor. This is the review's "documented loss against a qualified stronger baseline is a valid completion," now backed by numbers and a mechanism.
- **I3 (Opportunity 1):** the review asked to **attribute before changing**. Measurement is the attribution: hoisting the invariant weight-rounding is **~1.00x** at every M, so the compiler already eliminates it (or it is immaterial next to the GEMM + tile convert). Reverted to honor the stopping rule.

## Scope honored
No M-dispatch or serial/parallel-threshold tuning, no cross-op fusion, no engine integration, no end-to-end runs. F4 remains **PARTIAL** (shipped path still C1 + bit-exact I1; selftest OK), and standalone closure is **not** promotion — the continuous thresholds stay PROPOSED/UNRATIFIED and `promotion_gate.py` stays BLOCKED. The indexer M8 ~0.98x comparator caveat is unchanged and not relabelled. Lower-priority kernels (top-k, Sinkhorn, combine) remain provisionally closed; no change was made to them.

## Completion
Each of the three open kernel families now has a measured disposition for its justified opportunities across the full standalone M/shape/dtype (and, for S1, K) coordinates, with correctness evidence kept separate from speed. Compressor: C1 shipped (bit-exact win), C2 rejected, C3 not promoted. Indexer: I1 shipped, I2 qualified (conditional), I3 reverted (neutral). Sparse: fp32-bmm donor retained, S1 rejected. A failed candidate and a measured neutral are legitimate dispositions; no experiment was required to produce a win.
