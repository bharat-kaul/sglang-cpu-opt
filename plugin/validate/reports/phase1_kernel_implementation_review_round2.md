# Phase-1 Kernel Review: Round 2

2026-10-09. Independent reviewer: **GitHub Copilot**. Response reviewed at `c55b27f`, including implementation fix `2d4d223` and replicated launcher `8187a1f`, against reviewer commit `4d8edbf`. During review, HEAD advanced to `53d2a44`; that delta changes only workflow skills, not the reviewed runtime, benchmark, or response. Three read-only delegates and a parent full-evaluator probe supplied the evidence below.

## Decision

**CHANGES REQUIRED; retain the legitimate improvements, but do not sign off Phase 1.** The original indexer selection defect and dispatch-guard defect are fixed in the tested domain. The new indexer timing medians are correctly summarized. However, F4 still has executable hard-gate bypasses, sparse serving conformance remains pending, and the every-M no-regression requirement is unmet by the evidence. Several worthwhile optimization experiments remain untested; their benefit is neither proved nor ruled out.

This is a follow-up to [phase1_kernel_implementation_review.md](phase1_kernel_implementation_review.md), assessing [phase1_kernel_review.md](phase1_kernel_review.md). It does not ratify tolerances, modify production code, replace the preserved baseline, or authorize Phase-2 integration. Reporting PARTIAL for finite continuous approximation error is intentional; the structural defects below are separate from that policy.

## Closure Matrix

| Original finding | Disposition | Evidence / remaining boundary |
|---|---|---|
| P1-F1 indexer stage contract | **Closed, scoped** | Original signed FP4-grid counterexample now has zero logit error and zero non-tie mismatches at M1/8/16/32/64. Old arithmetic reinjection still fails. No full-model/GPU-wide certification is implied. |
| P1-F2 guards and empty batch | **Closed** | Dispatcher and tiled entry reject original malformed shape probes. Indexer N0 and compressor N0/D0 return correctly shaped empties. |
| P1-F3 F4 structural checks | **Partial** | Original malformed first-output tuples, all-masked reference and empty manifest now fail. Nonfinite indexer references, repeat/reference tuple structure, composed checks and inventory completeness remain defective. |
| P1-F4 sparse conformance/budget | **Partial, open** | PROPOSED labels and all-record aggregation improve. Part B matches operand rounding. F4 operands, output boundary, blockwise reference and downstream budget remain unresolved. |
| P1-F5 coverage | **Partial, open** | Signed indexer coverage now spans all M. Other required shapes/layouts and source-faithful sparse caller cases remain pending. |
| P1-F6 measurement | **Partial, open** | Three process-level indexer trials and median/spread are available. Same-contract comparator, complete run identity, other kernels' replication and every-M floor remain open. |
| P1-F7 GPU wrapper | **Exit masking closed; identity open** | Extracted real launch/exit block propagates stub child success/failure as 0/1. New metadata is not fully supplied or validated; device selection depends on what enumeration exposes. |
| P1-F8 overclaims / scale | **Partial** | Several report claims corrected. Contradictory claims persist; negative infinity still passes the new scale guard. |

## Residual Findings

### R2-F1: High - F4 Still Skips Mandatory Checks on Several Paths

These are residual instances of P1-F3, not evidence that every earlier fix failed:

- **Nonfinite indexer reference:** [../f4_acceptance.py](../f4_acceptance.py#L342) checks candidate finiteness, not reference finiteness, before selection. An actual compiled-kernel probe with one NaN in the query produced finite candidate logits and all-NaN oracle logits. Full `run()` returned **0/PARTIAL**, `non_tie=0`, `margin=nan`. NaN comparisons in the selection checker cannot validate membership. Reject invalid reference evidence before any metric or selection comparison, even if the caller domain should exclude that input.
- **Repeat/reference tuple structure:** [repeatability](../f4_acceptance.py#L317) and [tuple comparison](../f4_acceptance.py#L382) still use truncating `zip`. Full `run()` returned **0/PARTIAL** when the first candidate call produced three outputs but the repeat produced two, and when three candidate outputs were compared with an empty reference tuple. Validate both candidate executions and reference structure against the declared contract, not only the first candidate or the reference's shape.
- **Composed sparse bypass:** [the composed branch](../f4_acceptance.py#L295) returns before output shape/dtype, finiteness and repeatability checks. Parent fault injection through actual `run()` confirmed four cases: NaN output, FP64 output, an extra leading dimension, and an output changing on every invocation. Each returned **0/PARTIAL** with clean selection. There were exactly five candidate calls across five seeds, one per seed rather than a repeatability pair. Faults were injected into the candidate interface; this does not claim the native kernel ordinarily emits them.
- **Incomplete inventory:** [run()](../f4_acceptance.py#L413) now rejects an empty manifest, but one valid case with all other required cases removed returns zero. Nonempty cases with empty seed groups also return zero without evaluation. A complete gate needs an independently declared case/seed inventory, or an explicit restricted-scope result that cannot be mistaken for full qualification.

**Required closure:** establish one common hard-check layer for every result path, including composition: validate expected structure/dtype/device, permitted finite domain, reference validity, repeatability and coverage before family-specific comparisons. Retest the class through `run()`, including original controls and these sibling failures. Do not turn unratified continuous screening into a numeric hard gate merely to address structural defects.

### R2-F2: High - Sparse Comparisons Still Do Not Establish Serving Conformance

[Part B](../test_sparse_cpu_vs_gpu.py#L43) now feeds BF16-rounded operands to CPU candidates, a real improvement, but still compares FP32 CPU output to BF16 GPU output. [F4 sparse cases](../f4_acceptance.py#L245) continue supplying original FP32 candidate operands while the reference rounds to BF16. Keep that diagnostic distinct from a matched-operand, matched-output-boundary conformance check.

The source-faithful 64-entry recurrence and downstream budget are explicitly pending in the response; they remain gates, not newly discovered omissions. M1/8/64 shared-KV identity-index GPU evidence and M8/K512 F4 cases do not cover the requested independent batches, K128/160/640 unions, sentinels, offsets, TP heads or captured caller layouts. No new sparse measurement log beyond historical jobs384501/384502/384505 was located in the targeted search; the script fixes do not retrospectively rebind or recompute those runs.

[All-record aggregation](../f4_acceptance.py#L499) now requires every recorded row for a path to meet the proposed screen, including held-out records. That closes the earlier any-record bug, but does not ensure every expected record exists. The [aggregate maximum](../f4_acceptance.py#L503) mixes composed FP32-reference metrics into a quantity labelled versus the BF16 replica. Group metrics by reference identity and contract before aggregation. Continuous labels now correctly say PROPOSED/UNRATIFIED; no numeric budget is approved here.

### R2-F3: High - The New Run Does Not Establish a Same-Contract No-Regression Floor

Job384526 runs [../bench_idx_logits.py](../bench_idx_logits.py#L18), whose unchanged reference is FP32-stage arithmetic with nonnegative weights and a cosine-only check. It does **not** execute the corrected F4 signed-selection gate, assert correctness, or enforce a speed floor. The author's attribution of `non_tie=0` to this job is unsupported. Indexer correctness closure instead comes from the independently executed source-stage replay described below.

Three process-level trials improve the evidence, but every printed M1 pair is slower than Torch: **0.97/0.98/0.97x**. Calling that statistical parity is not justified by three ordered trials or the historical 33% noise statement. The tested comparator also performs different intermediate arithmetic; establish and time the actual same-contract fallback before using the ratio as a correctness-qualified replacement floor. A small absolute gap may be acceptable only through an explicit requirement change, not a relabelled result.

[The launcher](../run_indexer_replicated.sbatch#L14) records a short git ID, node, Torch version and thread count, but the log still begins `fatal: not a git repository`. Root discovery inside command substitution did not fail the overall launch. Actual worker affinity, dirty/source/binary/library digests and per-trial execution identity are missing; thread queries run outside `numactl`. Committed kernel/benchmark contents at recorded `8187a1f` match the response, which supports attribution but does not prove the built object or working tree used at runtime.

The median C++ and median printed-speedup rows are correct. The accompanying old off-roof factors are stale; corrected values are below. "The BF16 epilogue did not regress perf" is not established by cross-generation, cross-node comparisons. Other kernels still lack current source-bound replicated measurements.

### R2-F4: Medium - GPU Metadata and Device Selection Remain Incomplete

[../run_sparse_gpu_oracle.sbatch](../run_sparse_gpu_oracle.sbatch#L20) converts every enumerated GPU UUID into a container device argument, without intersecting the scheduler allocation. A mock with GPU1 assigned and two visible GPUs selected both. This is conditional on runtime visibility; it is not evidence that a real run used another allocation. Require allocation-aware selection rather than relying on enumeration to enforce it.

The wrapper does not forward `SLURM_JOB_ID` or `ORACLE_IMG` into the container. [The provenance stamp](../test_sparse_gpu_oracle.py#L97) therefore falls back to values such as `interactive` and a mutable `latest` tag, hashes a hardcoded source path after execution, permits `unavailable`, and does not verify the expected pin. CPU replay prints rather than validates that provenance. The current local kernel hash matches the independently fetched pin, but historical source/input/output/build identity remains unbound. Propagate the actual allocation/run/image identity, validate required hashes before execution, and enforce them on replay.

### R2-F5: Medium - Negative Infinity Still Selects Default Scale

[../../kernels/dsa_pilot/sparse_attend.cpp](../../kernels/dsa_pilot/sparse_attend.cpp#L15) uses `isfinite(scale) || scale <= 0`. Compiler-checked predicate results: NaN and positive infinity reject; **negative infinity accepts**. Preserve intentional finite default sentinels while requiring finiteness. This closes the nonfinite class rather than just the earlier NaN example.

### R2-F6: Medium - Completion and Exhaustion Claims Still Conflict With Evidence

The corrected preamble does not reconcile [the report's fusion/precision section](phase1_kernel_review.md#L114), which still claims no avoidable traffic, an irreducible sparse intermediate and resolved/ratified precision. [../results/kernel_opt_queue.json](../results/kernel_opt_queue.json#L88) retains stronger sparse conformance and completion claims; other queue entries retain DRAM-saturation and universal-floor wording. Existing provenance describes older validation/candidate stages. Preserve historical records, but label their generation and keep current qualification separate.

Indexer output is **FP32 storage containing BF16-rounded values**, not a BF16 tensor: both current C++ paths allocate FP32 and the top-k interface requires it. Its M1 path allocates a full 256-KiB score buffer at S1024/H64, disproving the blanket L1-only description, but not proving a DRAM round trip. Saturation and irreducibility require measurement; above-nominal useful bandwidth can also reflect cache traffic, accounting assumptions or timing uncertainty.

## Verified Indexer Fixes

Published model and kernel sources at HF pin `60d8d70770c6776ff598c94bb586a859a38244f1` were independently downloaded and SHA-256 matched to the local snapshot. The corrected [oracle](../f4_acceptance.py#L46) and [candidate epilogue](../../kernels/dsa_pilot/indexer_logits.cpp#L145) agree with the published stage boundaries in the tested domain.

- Exact prior seed-1 FP4-grid replay, H64/D128/S1024 and signed BF16-exact weights: **zero maximum error, zero non-tie mismatches** at M1/8/16/32/64.
- Reinjecting the old arithmetic yields **0.09235763549804688** error and the corrected full evaluator rejects **1/8/16/32/64** non-tie mismatches. This is a discriminating regression control.
- M1/tiled and FP32-KV/BF16-KV equivalence passed across the requested sweep; signed S1031 and strided M1/M8 checks also passed.
- **95 scoped manifest evaluations** passed hard gates; existing self-test passed. Exact-boundary tie alternatives passed; below-cutoff, omitted-above and duplicate selections failed.
- Original dispatcher/tiled malformed-shape probes now reject; empty-batch/channel probes return correct empty shapes. No FP32-storage adapter regression was found.

These are single-thread synthetic correctness probes, not proof over all finite inputs, multithread races, captured model shapes or full-model behavior. Persistent BF16-KV coverage is still only M1/32; compressor remains M8. Their scope must not be widened by the benchmark's M sweep.

## Performance Reconstruction

Raw log: `/scratch/bkaul/idx_replicated_384526.log`, trial rows7-11,15-19,23-27. Node `pcl-sprh08.sc.intel.com`, recorded git `8187a1f`, Torch `2.12.0+cpu`,64 threads. Each trial averages30 warm sequential calls, Torch first and C++ second, seed0 and reused buffers. Thus these are **medians of three process-level timing means**, not90 independent measurements. Ranges below are observed min/max, not confidence intervals. Ratios use separately rounded printed speedups.

| M | C++ median [min,max], us | Torch median [min,max], us | Speedup median [min,max] | Nominal ideal, us | C++ / ideal |
|---:|---|---|---|---:|---:|
| 1 | 100 [99,101] | 97 [97,98] | 0.97 [0.97,0.98] | 1.566 | 63.84x |
| 8 | 204 [203,205] | 618 [587,661] | 3.03 [2.86,3.25] | 12.531 | 16.28x |
| 16 | 217 [213,241] | 821 [738,862] | 3.85 [3.06,3.97] | 25.063 | 8.66x |
| 32 | 272 [272,281] | 924 [885,929] | 3.29 [3.25,3.42] | 50.126 | 5.43x |
| 64 | 395 [392,421] | 985 [985,1065] | 2.51 [2.50,2.53] | 100.251 | 3.94x |

Actual FP32 input/output storage implies useful bytes `561408M`; GEMM work is `16777216M` BF16 FLOPs plus `130048M` FP32 reduction operations and comparisons/conversions. Nominal anchors remain358.4 GB/s,124.5184 BF16 TF/s,7.7824 FP32 TF/s. Useful-byte time dominates this simplified bound. It omits dispatch, packing, allocation, executed intermediates and conversion costs; it is **not an achievable latency promise**. Direct BF16-KV would have53.440 us nominal ideal at M64, but job384526 does not time that contract.

M64's395 us is4.77% above the prior377 us observation at384482. Different intermediate arithmetic and nodes, unbound older builds and no paired trials prevent interpreting that as either a significant regression or proof of stability.

No new measurements for the other kernels were submitted. Historical C++ vectors below retain their original labels and limitations; they are **not newly replicated or requalified**. Order is M1/8/16/32/64, in us. Complete Torch comparisons and nominal ideals remain in the first report.

| Kernel | Historical C++ latency | Job |
|---|---|---|
| Sparse bestof, K512 | 66,147,247,416,749 | 384476 |
| Compressor R128/D512 | 25,79,141,271,555 | 384474 |
| Compressor R8/D512 | 16,20,25,28,42 | 384474 |
| Compressor R8/D128 | 13,14,14,16,20 | 384474 |
| Top-k | 2,12,13,13,14 | 384480 |
| Sinkhorn | 2,13,13,14,15 | 384480 |
| Combine | 2,12,11,12,14 | 384480 |

Sparse's M8 Torch comparator still varies320 to1336 us between historical jobs384474/384476. The9.08x ratio cannot be attributed wholly to fusion. Its whole-op FP32 utilization varies about13/47/56/66/74% across M. R128 compressor M64 useful bandwidth is about61 GB/s,17% of nominal, not uniformly7%. Sinkhorn/combine were already approximately this fast in the earlier C++ baseline; large Torch speedups are not newly demonstrated gains from this response.

## Optimization Opportunity Disposition

There is real implementation work, but the following opportunities have **not** been exhausted. A measured loss is a valid disposition; an untested claim of irreducibility is not. These are bounded next experiments, not predicted speedups. First repair correctness/measurement prerequisites; do not substitute experimental speed for acceptance.

| Priority | Implemented / evidenced | Remaining discriminating experiment |
|---|---|---|
| 1. Indexer | Parallel pack, tiled conversion, BF16-KV support and M1 specialization; published-stage correction verified | Conditionally omit unused [Abuf](../../kernels/dsa_pilot/indexer_logits.cpp#L118), then test scratch reuse. Compare whole-S and tiled M1 with identical stage precision and same-contract fallback; remeasure BF16-KV under the corrected kernel. |
| 2. Sparse | FP32 donor plus fused in-place softmax is a legitimate implementation improvement; naive BF16-AMX variant tested and rejected | Resolve comparator instability; test source-faithful blockwise BF16-input/FP32-output library GEMMs on matched operands/output boundaries and K128/160/640. The rejected variant does not rule this out. |
| 3. MHC | Existing fused Sinkhorn recurrence and accumulate-once combine outperform Torch in historical logs | A/B serial versus parallel grain at every M using a fixed pool. Caller fusion remains Phase-2 work under the agreed order; do not advance phases to test it prematurely. |
| 4. Compressor | N-by-D tiling and all three historical shape sweeps; large R128 M1 gain | Compare vector-exp or stable multipass pooling against two exponentials per recurrence step, preserving FP32 state/masks. Prioritize R8 by amortized call weight. Author-reported BF16-input slowdown lacks a located raw record and does not eliminate native-donor alternatives. |
| 5. Top-k | Serial M1 and guarded chunking remove demonstrated small-M overhead | Test reusable scratch and captured S/k dispatch. S1024/k512 does not exercise useful chunked selection; preserve exact non-tie membership and tie handling. |

Conditional M64 standalone call-weighted costs from the existing model are8.295 ms indexer,1.290 ms Sinkhorn,1.204 ms combine,0.412 ms compressor and0.294 ms top-k. Compressor splits into0.087/0.221/0.105 ms for the three shapes. These are opportunity sizes, **not E2E measurements or promised savings**. Sparse cannot be priced from a single K512 fragment against heterogeneous real layer unions. Higher-ranked experiments deserve evidence first; every imaginable micro-optimization need not be implemented.

## Required Next Submission

1. Close the common F4 hard-check class across every branch and prove it with full-entry fault injection, including expected coverage and seed inventory. Retain the verified indexer contract/guard fixes.
2. Supply source-faithful sparse conformance at identical operand/output boundaries and the required captured cases. Keep downstream budgets pending until independently justified and ratified.
3. Bind source/build/library/affinity/input identity; time a conformed fallback, use at least three independent paired trials with order variation, report medians/spread, and disposition M1 indexer and M8 sparse comparator behavior. Preserve all required M values and the baseline as a separate historical record.
4. Record the highest-ROI untested experiments above as measured win/loss or explicitly deferred with scope and rationale. Reconcile current report/queue/provenance claims. Do not claim universal no-regression, saturation, precision resolution or exhaustion from the present evidence.

## Review Limits

No cluster/GPU jobs, model loads or new performance runs were launched. Correctness used `/scratch/bkaul/venvs/sglang-cpu/bin/python`, one Torch thread, `OMP_NUM_THREADS=1`, `MAX_JOBS=2`, C++ flags `-O3 -fopenmp -march=native` where compiled probes were needed. The parent composed probe used Pylance's explicit-interpreter runner and requested no file outputs; delegates exercised the other stated checks. Sparse wrapper/device tests used stubs, and its infinity predicate used a small compiled check, not a cluster launch. Only this report is authored. The new self-audit instructions are acknowledged, not audited as a separate customization task or treated as evidence that runtime gates now pass.