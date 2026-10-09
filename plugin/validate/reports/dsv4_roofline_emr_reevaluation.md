# DSv4 EMR Roofline Re-evaluation

Date: 2026-10-08.

Reviewed revision: `3535a08b9452372c3af70c210c58966e87741069`, compared with the [previous verdict](dsv4_roofline_emr_verdict.md) committed in `8bd8c18`.

Reviewed artifacts:

- [Roofline report at the reviewed revision](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/reports/dsv4_roofline_emr.txt).
- [Generator at the reviewed revision](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/dsv4_roofline_p2.py).

This is a revision-specific analytical review, not a claim about changes made after that commit. No new hardware benchmarks were needed or run for this re-evaluation. It does not modify the generator, platform profiles, kernels, or earlier verdict.

## Verdict

**The changes are locally valid but substantially incomplete. The revised roofline model should not yet be accepted as quantitatively correct.**

Only three generator lines changed after the previous verdict, updating three `OPS` entries. Corresponding report rows were regenerated. The report body reproduces exactly from the revised script, excluding its introductory header: this is not a stale-report problem. The remaining defects are in the generator's accounting.

Keep the agreed **machine-peak targets**. Neither sustainable-ceiling calibration nor full SGLang integration is a prerequisite for correcting these issues. An optimistic but coherent workload model is acceptable; incorrect shapes, missing work, and contradictory byte accounting are not.

## Remaining Findings

### 1. High: Attention Batch Traffic and Fusion Accounting Remain Incorrect

[The attention formula](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/dsv4_roofline_p2.py#L103) still multiplies FLOPs by `M` without multiplying KV-cache traffic by `M`. For independent requests, the KV term needs `M * effective_context * HD * kv_bpw`, with explicit reuse and dtype assumptions.

It also includes a score-sized `M * NH * Sctx` byte term despite describing flash fusion. Independent-request batch scaling and the reported compute-bound transitions therefore remain unsupported. An explicitly shared-prefix workload would require different accounting, not an implicit cache shared across arbitrary requests.

Executing the current functions gives full-attention bytes of 115,539,968 at M=1 and 293,076,992 at M=8, rather than eight times the M=1 count for the same independent-request work. Its AI rises from 199.8 at M=1 to 862.3 at M=64. The sparse-attention function has the same ownership problem. These checks address the declared ideal traffic model, not measured cache behavior or the current masked shared-pool fallback.

### 2. High: The Shared Expert Still Counts Only One Matrix

[The shared-expert entry](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/dsv4_roofline_p2.py#L156) still calls the single-GEMM helper. Setting `fused=True` changes descriptive text, not the number of matrices.

At M=1, under the script's own dimensions and 41-layer count:

- Counted matrix FLOPs: **687,865,856**.
- Gate/up/down matrix FLOPs: **2,063,597,568**.
- Reported time: **0.481 ms**.
- Three-matrix weight-only bound at 358.4 GB/s: **1.439 ms**.

Replace this entry with the complete three-matrix expert cost. Fusion cannot eliminate the gate, up, or down weights.

### 3. High: Capacity Was Not Updated With the Revised Projections

[The `p0()` formulas](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/dsv4_roofline_p2.py#L175) retain the original dimensions, while `OPS` now uses the corrected `wo_b` and indexer-query dimensions and an explicit `weights_proj`.

| Weight category | `p0()` capacity | Revised `OPS` weight inventory |
|---|---:|---:|
| MLA projections | 3.336569 GB | 4.599054 GB |
| Listed indexer projections | 1.363149 GB | 0.356516 GB |

These values compare the generator with itself, using its currently declared dtypes and listed GEMMs. They do not represent complete corrected capacity: other operations, grouping, actual storage formats, scales, and runtime memory still require accounting.

Derive capacity and per-op costs from the same tensor inventory. The report currently contradicts itself even before those additional corrections.

### 4. High: Ridge-Crossing Annotations Remain Wrong

[`cross_M`](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/dsv4_roofline_p2.py#L61) still uses the weight-only approximation while the AI table includes activation bytes. Solving the actual byte formulas at the report's nominal ridge gives:

| Revised projection | Reported crossing M | Crossing from its byte formula |
|---|---:|---:|
| `wo_b`, 8192 -> 4096 | 174 | **199.19** |
| Indexer query, 1024 -> 8192 | 174 | **281.25** |
| `weights_proj`, 4096 -> 64 | 348 | **None: AI saturates at 63.02** |

The newly added `weights_proj` row therefore introduces another instance of the same contradictory annotation. Its narrow output dimension prevents it from reaching AI=348 under this traffic model, regardless of batch size.

Use `denominator = flops_per_token - RIDGE * bytes_per_token`; return no crossing when that denominator is nonpositive, otherwise use `RIDGE * fixed_bytes / denominator`.

### 5. High: The Fused Indexer and Operation Graph Remain Incomplete

[`indexer_fused`](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/dsv4_roofline_p2.py#L112) still omits batch scaling of keys and explicit query/head-weight reads while retaining intermediate-score traffic. The report consequently does not model the fused operator it names.

Raw context 4096 still substitutes for compressed index context. For ratio-4 indexing, use the appropriate available compressed positions, approximately 1024 at that raw context before masking/availability details. Adding `weights_proj` does not account for compressor key/score projections, normalization, RoPE, or Hadamard work.

The operator list still combines full-context and sparse attention without resolving the intended sliding-window/compressed execution graph or verifying invocation counts. These omissions cannot be repaired by correcting projection dimensions alone.

### 6. Other Requested Corrections Remain Unapplied

- `wo_a` is still modeled as an ungrouped FP8 GEMM with 1024 outputs, rather than the eight-group shape yielding 8192 outputs. Current CPU BF16 storage and a prospective FP8 target remain conflated.
- Expert occupancy still predicts **5.941710533 distinct experts for one top-6 request**. Under uniform independent requests with distinct top-k selections, use `E * (1 - (1 - TOPK/E)**M)`.
- Quantization-scale metadata and layout overhead remain omitted from storage and traffic.
- Fixed norm/top-k/compressor/Sinkhorn latency constants remain unexplained machine floors. Zero FLOPs and zero bytes do not establish latency as the binding resource.
- Hardware-profile provenance, layer counts, the hash/MHC distinction, capacity overheads, GB/GiB units, and consistent per-call/per-step/fused totals remain unresolved.

These findings do not require replacing nominal targets with achievable targets. They require consistent definitions and justified accounting.

## Valid Changes

The three changed [`OPS` entries](https://github.com/bharat-kaul/sglang-cpu-opt/blob/3535a08b9452372c3af70c210c58966e87741069/plugin/validate/dsv4_roofline_p2.py#L142) are useful corrections. Their revised FLOP and byte calculations were checked by executing the actual functions:

| Entry | Revised dimensions | M=1 time under its declared model |
|---|---|---:|
| `wo_b` | 8192 -> 4096 | 4.028731 ms |
| Indexer query | 1024 -> 8192 | 938.285714 us |
| Explicit `weights_proj` | 4096 -> 64 | 59.442857 us |

The `weights_proj` timing uses its declared BF16 storage assumption. These are arithmetic validations of the revised rows, not measured kernel-performance claims or certification of the complete operation graph.

## Status Against the Ten Corrections

| Previous correction | Status in the reviewed report and generator |
|---|---|
| 1. Separate hardware profiles | Not addressed |
| 2. Correct output projections | Partial: `wo_b` fixed; `wo_a` grouping/storage unresolved |
| 3. Correct indexer projections and missing work | Partial: query shape fixed and `weights_proj` explicit; remaining work omitted |
| 4. Shape-dependent ridge crossings | Not addressed |
| 5. Attention ownership and fused traffic | Not addressed |
| 6. Effective contexts and invocation counts | Not addressed |
| 7. Consistent fused-indexer accounting | Not addressed |
| 8. Three-matrix shared expert | Not addressed |
| 9. Expert occupancy and quantized storage | Not addressed |
| 10. Latency models, capacity, and consistent totals | Not addressed |

Newer kernel benchmarks elsewhere in the branch do not repair these analytical formulas and were not re-evaluated as part of this review.

## Verification and Next Revision

Verification performed: compared the two requested files against `8bd8c18`; regenerated the report and matched its body exactly; executed checks for the three dimension changes, attention/indexer batch scaling, shared-expert matrix count, ridge crossings, capacity consistency, and distinct top-k occupancy.

The next revision should correct the accounting functions and derive capacity and per-op costs from one inventory, then test those invariants before regenerating the report. Retain the previous verdict's tracking guidance: prioritize latency and useful throughput, track useful versus executed FLOPs and traffic separately, and use achieved bandwidth/FLOP/s and distance from the nominal roof diagnostically. Recompute AI when work or traffic changes.

**Recommendation: accept the three local projection corrections, but do not treat this revision as having implemented the full roofline feedback.**