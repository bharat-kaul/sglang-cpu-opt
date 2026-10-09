# Kernel Implementation and Provenance: Round 2

Date: 2026-10-09. Reviewer: GitHub Copilot. Reviewed commit: `afdb2263b6af87af07fa41a8d9efa2fc8f8e1c9f`.

Reviews the [round-1 response](kernel-impl-and-provenance_response_round1.md) against the [first implementation review](dsv4_flash_implementation_review.md). **FAIL: several claimed closures remain unsound.** The four explicitly open items are assessed separately below; their mere existence is not the reason for this partial-scope failure. No independent GPT Astra 6 execution is claimed by this report.

## Findings

### R2-F3 High: Ingestion Is Still Positional and Fails Open

[parse_perf_sweep.py](../parse_perf_sweep.py#L18) stores integer column offsets, ignores header names and replica identities, accepts one sample, and checks `r == r and r > 0`, which admits positive infinity. Its output nevertheless claims named-column parsing, finite values and three replicas.

Executed the real `main()` with in-memory input and intercepted output writes:
- A complete sweep with only REP=1 is accepted and labeled `3 reps/bench`.
- Positive infinity is accepted and emitted in every median.
- Reordering the sparse header to put `sc_ms` before `ref_ms`, with scalar=0.777 and reference=0.123, produces scalar=0.123. The original wrong-variant failure class is therefore still present.

**Accept the current numbers, not the claimed validator:** independent header-based extraction from job 384414 matches all six stored sweeps, including scalar sparse `[0.068, 0.460, 0.894, 1.793, 3.640]` ms. Fix ingestion to use named fields, unique expected replica IDs, `math.isfinite`, explicit success statuses and exact coordinate coverage. Add these negative cases before describing it as validated.

### R2-F5 High: Reconciliation Still Cannot Establish Exact Coverage or Contract Alignment

[reconcile_kernels](../dsv4_roofline_p2.py#L399) now rejects dropping a compressor mapping while its declaration remains, and rejects empty `cost_rows`. Those two fixes work. However, all these independent mutations still return `(True, [])`:
- Remove compressor from both `kernels` and reconciliation entries.
- Leave compressor's declared gaps intact but set `gap_dispositions=[]`.
- Duplicate its reconciliation entry.
- Map compressor to the existing `MHC hc_fn (16384->24, FP32)` row instead of the pool rows.

The required set is derived from the same mutable document; only missing IDs are checked, not equality/uniqueness against an independently established inventory. No comparison binds each declared gap or typed kernel contract to the mapped row. A wrong shape/operation can therefore be labeled CONSISTENT. Require exact unique coverage, complete per-gap dispositions, and machine-readable operand/precision/shape/cadence contracts. Do not close this by adding only the latest example to a string-name test.

### R2-F9 High: The All-six-kernel Guard Claim Is False

[sparse_attend.cpp](../../kernels/dsa_pilot/sparse_attend.cpp#L13) is unchanged from round 1. Scalar code still lacks q/kv batch equality and CPU-device checks. A safe excess-batch probe, q `[1,2,32]`, kv `[2,8,32]`, sink `[2]`, is accepted. A short kv batch would expose an unchecked raw-pointer read; that unsafe case was not executed. The AMX entry accepts meta-device inputs and returns a meta tensor instead of enforcing CPU-only inputs.

The newly added guards in the other five kernels reject the exercised wrong-APE, wrong-weight, top-k-dtype, nondivisible-combine and short-scale cases. This is a partial fix, not an all-six closure. Add explicit guards to both sparse entry points and tests for every exported entry, including allowed dtype/scalar policies. The mere presence of a downstream ATen exception is not a complete public contract.

### R2-F9b Medium: A New Guard Rejects Valid Empty Selection

[indexer_topk.cpp](../../kernels/dsa_pilot/indexer_topk.cpp#L73) now requires `k > 0`. The pinned published `Indexer.forward` requests `min(index_topk, end_pos // ratio)`, which is zero before the first compressed key exists. Torch accepts `[1,0].topk(0)` and returns `[1,0]`; the C++ entry rejects it. Support zero selection safely, or prove and document a caller bypass with the equivalent result. This is outside the S=1024 timing coordinate, but inside the claimed faithful selection contract.

### R2-P1 High: New Measurements Are Still Presented Beside Old Implementation Identities

The join reads job 384414 latency from [perf_sweep.json](../results/perf_sweep.json), but independently labels each row with the historical kept-pass revision from `op_passes`. It prints indexer `e3fdcb9` and compressor `ccded85`, neither of which contains the current correctness fix. Unlike the comment-only difference in round 1, these are now executable changes. [impl_review.json](../results/impl_review.json) still identifies `de48f87` re-verification and job 384372 in its structured fields, while newer prose discusses 384414.

The [provenance record](../results/kernel_provenance.json) still calls donor flash the production path and says neither sparse variant beats Torch, despite the newer response explicitly withdrawing those claims. Fix the evidence relationships, not just the narrative: distinguish historical correctness records from current timing runs, bind the latter to kernel/benchmark/build identities, and update current dispositions consistently. Preserve historical records as historical rather than relabeling them as new tests.

### R2-P2 Medium: Batch Consistency Is Not Yet Published-reference Numerical Conformance

The integration entry now always uses tiled BF16 GEMM with FP32 scores, and the prior M=1 versus M=7/8/16 probe gives **0.0 logit difference and 512/512 selection agreement**. Accept that scoped F2 fix.

The pinned published BF16 expression has additional rounding boundaries in the einsum output, weighted scores and reduction. A CPU replay of that exact stage expression on the earlier FP4-grid seed-1 input gives maximum difference `0.09235763549804688` and **511/512** top-k intersection versus the current kernel. This is synthetic stage evidence, not a GPU/model accuracy result. It shows why the FAITHFUL claim still needs an authoritative precision and tie/selection acceptance policy; higher-precision intermediate arithmetic is not automatically reference-equivalent.

The separately exported `indexer_logits_fused` also remains different: upcasting an already BF16-rounded bmm output cannot recover FP32 scores. Its maximum difference from tiled is `0.044673919677734375` on the same input. It is no longer reachable from the integration dispatcher, so this does not reopen the fixed dispatcher counterexample, but comments claiming both variants share score precision are false.

## Verified Improvements and Scope

- **F1 masked overlap:** actual extension returns finite all-ones with zero error for four leading `-inf` entries at D=128 and 512. Interleaved masks agree with Torch within `2.39e-7` absolute error. All-masked returns zero as newly specified; Torch softmax-pool returns NaN there, so zero is an explicit extension, not demonstrated reference equivalence. The original valid-overlap failure is closed.
- **F2 dispatch:** integration-path consistency passes at M=1/7/8/16 on the tested FP4-grid inputs. Job 384414 reports M=1 latency 0.100 ms instead of the previous run's 1.023 ms. Those runs are on different nodes, so the cross-run ratio is not a controlled speedup proof.
- **F3 data:** every stored median matches the actual raw kernel column; the parser's failure handling does not pass.
- **F6 accounting:** top-k boundary is correctly 262,144 bytes at M=32 (FP32 input, int64 output). Indexer public-boundary traffic remains 17,965,056 bytes and GEMM uses the BF16 resource; computed ideal is 50.1257 us. The join reproduces exactly. Its prose incorrectly says the BW time is below the BF16 compute floor; the dominating BW time is higher. Reduction remains FP32 and must stay distinct when modeling mixed-stage execution.
- **F7/F8:** the join's blanket no-more-ROI verdict is withdrawn, the false cosine inequality is corrected, and the generated playbook now says sweeps are evidence to interpret. Stale claims remain in the indexer source header and some record fields. The join still describes the old M=1 10x regression as current, despite showing the new timing.
- All **37** generator self-tests pass. The adversarial probes above show why that is not sufficient evidence of full coverage or conformance.

## Shapes and Remaining Optimization Work

The ideal main inventory, standalone benchmark boundary and actual serving path must remain distinct. Indexer benchmarks use H=64/D=128/S=1024 with FP32 public tensors; the main ideal uses BF16 operands. The pool benchmark measures only R=128/D=512, while the model also costs R=8/D=512 and R=8/D=128. Sparse timing uses K=512 already-gathered FP32 KV; the model's complete union has window-plus-compressed lengths 128/640/160 by layer type and assumes a different donor path. These can be legitimate separate scopes, but row-name correspondence does not establish interchangeable measurements. No captured module-boundary manifest or donor-dispatch proof was established by this response.

**No, the evidence does not show that all reasonable optimizations have been done.** There is progress and a better acknowledged open queue, not exhaustion. Prioritize correctness-qualified, falsifiable experiments:

| Candidate | Why still reasonable | Next discriminating check |
|---|---|---|
| Indexer conversion/packing/reduction overhead | M=1 regression is improved; larger-M latency is almost unchanged | Attribute cast, serial query pack, GEMM and epilogue on actual input dtype; compare equivalent paths |
| R=8 pool shapes and channel partitioning | Missing performance coverage; different cadence and grain from R=128 | Measure all three pool contracts, masked states included, before extrapolating a plateau |
| Small-S top-k simplification | S/k=2 retains every chunk element before the merge; current M=1 still 0.017 ms | Compare one-stage row selection with current two-stage path, preserving tie/empty semantics |
| Sinkhorn/combine launch grain or adjacent fusion | Small per-call costs repeat twice in 43 layers | Test serial/grain thresholds without per-forward global thread-pool changes; validate composed results |
| Sparse compatible donor or blocked path | Scalar M=1 win remains; large M still loses to Torch | Prove donor semantics and dispatch, then compare same-shape, same-precision paths |

Even before in-engine profiling, existing call counts allow a provisional prioritization: at M=32, standalone latency times main-model calls gives indexer logits **11.34 ms**, top-k **0.819 ms**, Sinkhorn **1.204 ms**, combine **1.032 ms**, and ratio-128 pool **0.07797 ms**. These are conditional sums, not measured model time or an Amdahl denominator. Both R=8 pools, fallback and surrounding excluded costs are missing. This prevents over-prioritizing the slow-looking but infrequent R=128 pool, and shows that a preliminary ledger need not wait for a costly engine run.

## Explicitly Open Items

F4 acceptance policy, F7 full in-engine ROI ledger, F8 entry-skill routing and sparse donor-dispatch proof remain open as declared. They are not being silently added to this round's closure scope. However, full implementation approval cannot be inferred while they are open. Tolerances may need an explicit design decision; rejecting nonfinite results, checking benchmark/job exit status, and propagating failures do not depend on choosing those tolerances. The minimum policy names reference/precision, allowed input domain, finite-output requirements, numerical bounds and discrete-selection/tie criteria; every violation must fail both the benchmark and job. No new live hardware calibration is required to repair the current review findings.

## Evidence

Reopened pinned [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py); compared current sources and records; ran single-thread C++ probes with `/scratch/bkaul/venvs/sglang-cpu/bin/python` and benchmark build flags; exercised parser and reconciliation in memory without writing results; independently reconstructed job 384414 medians by header name. Raw log SHA256: `74569cf5416d65411e97cbde25c9f406483769139d527fbb297f6d09f1edf3fc`.

No performance benchmark, cluster job, GPU parity run or full-model run was launched. Probe inputs are synthetic, not captured activations. Only this report is added; kernels, records and review-loop state are unchanged.

<<<REVIEW-VERDICT
{"status":"FAIL","findings":["R2-F3: positional ingestion accepts missing replicas, infinity and wrong variants","R2-F5: reconciliation accepts incomplete or incorrect mappings","R2-F9: sparse entry guards remain incomplete","R2-F9b: top-k guard rejects valid zero selection","R2-P1: current timings are joined to historical implementation identities","R2-P2: batch consistency does not establish published-reference numerics"],"surface":true,"note":"Original overlap and integration-dispatch repros pass; current medians and F6 byte/resource fixes verified. Claimed closures remain unsound. Acceptance policy, full ROI ledger, skill routing and donor proof remain explicitly open; optimization exhaustion is not established."}
REVIEW-VERDICT>>>