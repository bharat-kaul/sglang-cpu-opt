# DSv4 Flash Roofline Gate: Re-review

Date: 2026-10-08.

Reviewed revision: `6eb2f68296efd68e3197db6a1078e6e9a350ab1a`.

Responds to the [author response](dsv4_flash_roofline_gate_response.md), following the [gate review](dsv4_flash_roofline_gate_review.md).

## Verdict

**FAIL: substantial corrections verified, but the roofline correctness gate remains open. Implementation review remains deferred.**

The major attention-graph, main-layer-count, shared-expert-storage, and MHC-matrix-dimension corrections are real. All 13 current self-tests pass. The main report and the new EMR join report regenerate exactly. Remaining problems are in accounting and reporting, not stale generation of those two artifacts.

The response overstates R2, R4, R5, and R7 as fully fixed. In particular, a single-point annotation does not restrict the executable batch sweep; adding some byte terms does not establish correct operand ownership; and changing a precision label does not change the compute ceiling.

Retain nominal machine-peak targets. New hardware calibration, full-model execution, and implementation benchmarking are not required to resolve the findings below. Explicitly excluded work may remain excluded from a clearly scoped partial model; it must not be hidden inside apparently complete rows or totals.

## Verified Corrections

Independent comparison with the [pinned checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) and [reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py), plus execution of the revised accounting functions, confirms:

| Quantity | Verified current result |
|---|---:|
| Main-layer counts, ratios 0 / 4 / 128 | 2 / 21 / 20 |
| Attention positions summed over 43 main layers | 16,896 |
| Attention QK/AV FLOPs, M=1 | 2,214,592,512 |
| Attention bytes, M=32, declared BF16 model | 734,003,200 |
| Attention nominal BW time, M=32 | 2.048 ms |
| Main compressor projection matrix bytes | 520,093,696 |
| Indexer compressor projection matrix bytes | 88,080,384 |
| Shared-expert FP8 matrix/scale bytes | 1,082,196,480 |
| MHC hc_fn matrix FLOPs, M=1 | 67,633,152 |

Compression shapes and average invocation counts now distinguish ratio-4 overlap, ratio-128 pooling, and the separate indexer compressor. The matrix inventory includes the previously missing compressor projections. Shared experts use FP8 while routed experts remain MXFP4. MHC hc_fn uses K=16384 with FP32 parameter bytes. These fixes should be preserved.

The hardware profile now distinguishes the original ISA scan from the measurement node, removes the stale recoverable-headroom claim, and uses the correct **124.5184 TFLOP/s nominal BF16 reference**. No new hardware measurements were performed or certified in this re-review.

## Remaining Findings

### F1. High: Single-Point Measurements Still Populate Every Batch Column

The [measured helper](../dsv4_roofline_p2.py#L111) stores a prose note, but [Op.row](../dsv4_roofline_p2.py#L93) unconditionally returns the same latency for every M. The [renderer](../dsv4_roofline_p2.py#L307) does not print `op.note`, so the main emitted report contains no `single-point` annotation at all.

Executed results, per step:

| Row | M=1 | M=32 | M=64 | Available annotation in source |
|---|---:|---:|---:|---|
| Indexer top-k | 819 us | 819 us | 819 us | M=32 only |
| MHC Sinkhorn | 1,290 us | 1,290 us | 1,290 us | M=32 only |
| MHC combine | 1,118 us | 1,118 us | 1,118 us | M=32 only |
| q_norm | 215 us | 215 us | 215 us | `M=unmodeled` |

The numbers remain under an IDEAL roofline heading despite being observations or unsupported placeholders. Norm/RoPE/embed/hash values still lack an identifiable measured operating point and source result. The tracker still describes them as approximate small-op floors.

The join's new provenance block also identifies an upstream model revision, not the CPU benchmark implementation revision or the raw result from which each measurement was selected. A model SHA cannot supply missing measurement provenance.

**Required closure:** represent measured coordinates as structured data; render only at matching coordinates, otherwise N/A; print provenance or provide an auditable result link. Move measurements to an observation table or unambiguously distinguish them from ideal targets. Values without an identifiable observation must be explicitly unmodeled, not populated as measured constants. Assert this behavior in a renderer test.

### F2. High: New Pooling Rows Omit the Score Stream and Misstate State Precision

The new [pool helper](../dsv4_roofline_p2.py#L118) counts `(win*D + D)*2` bytes per request: one input stream plus one output. Softmax pooling consumes both KV values and scores. The [reference compressor state](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L303) stores both streams as FP32; its boundary operation reads both to compute the weighted pool. No score-generation fusion or score-residency exception is declared by this helper.

Bytes per boundary call at M=32, before multiplying by average calls/step:

| Pool | Current model | Two BF16 streams + BF16 output | Two FP32 streams + FP32 output |
|---|---:|---:|---:|
| Main ratio-4, win=8, D=512 | 294,912 | 557,056 | 1,114,112 |
| Main ratio-128, win=128, D=512 | 4,227,072 | 8,421,376 | 16,842,752 |
| Indexer ratio-4, win=8, D=128 | 73,728 | 139,264 | 278,528 |

The latter two columns illustrate explicit tensor-boundary conventions, not competing measurements. The reference pool produces an FP32 intermediate before the subsequent dtype conversion/normalization. Fusing that conversion and writing BF16 changes the output term, not the need to read both FP32 state streams. Even the hypothetical all-BF16 variant has almost twice the current traffic.

The reference also performs per-token state writes and compression postprocessing. The newly modeled GEMM outputs are charged as BF16 by the generic helper, while the reference state is FP32. Checkpoint-BF16 projection weights do not imply BF16 projection outputs or state. The tracker exposes indexer normalization/RoPE/Hadamard omissions, but does not fully describe the main compressor's remaining state/postprocessing boundary.

**Required closure:** inventory KV state, score state, and output separately, with explicit dtypes and memory-hierarchy assumptions. Reconcile projection outputs/state writes with pool reads. Either model omitted state/postprocessing work or list it as excluded. Keep the corrected cadence and projection matrix capacities; do not revert to a fixed pooling latency.

### F3. Medium: FP32 Labels Still Use the BF16 Compute Model

Every non-latency row in [Op.row](../dsv4_roofline_p2.py#L93) uses the same BF16 AMX `PEAK` and `RIDGE`; `prec` does not select a compute resource. This includes the newly FP32 MHC rows. An execution probe with a 4096x4096 GEMM at M=4096 returns the identical **1.103764 ms** compute time for `prec='bf16'` and `prec='fp32'`, both calculated from the BF16 AMX peak.

This is a controlled helper probe, not a claim that actual MHC has that GEMM shape. The published MHC rows are BW-branch dominated under the existing model, so this probe does not establish that their displayed times must change. It does establish that the claimed precision-conformant compute model has not been implemented.

[hc_fn](../dsv4_roofline_p2.py#L172) also uses `(K+N)*AB` with `AB=2` for activation traffic. The [reference hc_pre](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L676) casts its flattened input to FP32 and computes FP32 mixes. An explicitly fused BF16-input/FP32-internal path could avoid a materialized FP32 input read, but the output and fusion boundary still need separate accounting. Changing only parameter bytes does not establish that boundary.

**Required closure:** separate stored dtype, operand dtype at the modeled boundary, and compute dtype. Use a justified nominal FP32 compute reference, or explicitly report a DRAM-only partial bound with the FP32 compute term unmodeled. A prospective BF16 compute path must be labeled as such. Add a test that precision affects resource selection or produces the explicit unsupported/unmodeled state. This does not require measured FP32 calibration.

### F4. Medium: The Join Adds Incorrect Operands and Still Omits Required Ones

The revised [join inventory](../dsv4_roofline_vs_measured.py#L32) does not yet match its declared standalone boundaries:

- **Indexer:** the added `64*128*2` term is called `wq_b head query weights`, but the logits operation consumes an already projected query. That is neither an input to this boundary nor the actual 1024-to-8192 projection matrix. Head weights are computed from each request, so the term is `M*64*4`, not `64*4` shared across M. The main [indexer model](../dsv4_roofline_p2.py#L139) still uses FP4-sized query bytes on the grounds of FP4 simulation, while the join uses BF16. The reference's in-place simulation does not establish packed query storage.
- **Compressor:** the per-request output term is fixed, but the prior positional-score operand was removed. The [benchmark call contract](../bench_compressor.py#L31) passes `kv`, `score`, and `ape`, with `ape[R,D]`. For this standalone boundary, count that read once under reuse, or explicitly define a different pre-added-score boundary and do not attach its measurement to it.
- **Sinkhorn:** only the 4x4 plan output is counted. The [benchmark call contract](../bench_sinkhorn.py#L30) takes mixes, scale, and base, and returns pre, post, and combination tensors. Pre/post writes and scale/base reads are missing.
- **Combine:** its weights are `pre[M,4]`, as shown by the [call contract](../bench_combine.py#L35), not a single shared `[4,4]` matrix. The arithmetic still omits reduction additions: `M*H*4` counts multiplies, while a four-term weighted sum requires `M*H*(4+3)` multiply/add operations.

At M=32, FP32 for the specified small-op operands, and BF16 query/keys for indexer:

| Standalone boundary | Current bytes | Bytes from the stated call contract |
|---|---:|---:|
| Indexer logits | 9,060,608 | 9,052,160 |
| Compressor pool, including shared APE | 16,842,752 | 17,104,896 |
| Sinkhorn, including all outputs and scale/base | 5,120 | 6,252 |
| Combine, including per-request pre weights | 2,621,504 | 2,621,952 |

Combine FLOPs are **524,288** currently versus **917,504** for the weighted sum. Main indexer query accounting gives **8,667,136 bytes per call** at M=32, versus **9,052,160** with BF16 query storage. These differences vary in importance, but the larger issue is inconsistent operator boundaries, not a rounding tolerance.

The join still concludes that a large gap proves overhead dominance and confirms donor routing. Roofline distance alone proves neither. The existing observations can remain observations; interpreting their causes and validating donor implementations belongs to the later review phase.

**Required closure:** derive each row from its actual input/output contract and explicitly declared fusion. Correct request ownership and precision. Recompute the join, retain a link to each measurement source, and remove causal conclusions unsupported by this analytical review. No kernel changes are requested here.

### F5. Medium: Tracker Failures Bypass the Publication Gate

The [tracker reader](../dsv4_roofline_p2.py#L248) catches every exception and silently continues. Injecting `FileNotFoundError` returns only the two locally annotated MHC exclusions; all tracker-only exclusions disappear. The conformance self-test neither loads the tracker nor checks provenance/measurement coverage.

The new hard-coded reference expectations are useful regression tests and agree with the pinned config at this revision. They are not a blanket conformance guarantee: changing a pooling byte formula, leaving unsupported measured coordinates populated, or supplying an invalid tracker does not fail them. Requiring network access on every self-test is unnecessary; pinned local fixture data and contract assertions are sufficient.

There is also a source-provenance typo: the full revision recorded in the [generator header](../dsv4_roofline_p2.py#L6), [join header](../dsv4_roofline_vs_measured.py#L10), and [tracker](../results/roofline_open_items.json#L3) is `60d8d770770c6776ff598c94bb586a859a38244f1`, not the verified `60d8d70770c6776ff598c94bb586a859a38244f1`.

**Required closure:** fail report emission on missing, malformed, or invalid tracker data; enforce valid dispositions and coverage; test single-point rendering, score-stream accounting, per-request weights, and precision selection. Centralize the correct reference identifier. Preserve explicit partial-model exclusions instead of treating their presence as a claim of completeness.

## Disposition Against R1-R7

| Previous finding | Re-review disposition |
|---|---|
| R1: Attention graph/main-layer counts | **Resolved for the declared 4096-token main-model point.** Correct sources, counts, FLOPs, and bytes verified. |
| R2: Compressor subgraph | **Partly resolved.** Matrix inventory, shapes, and average cadence fixed; state/score traffic and remaining boundary exclusions unresolved (F2). |
| R3: Shared-expert precision | **Resolved for analytical storage accounting.** FP8 matrices/scales and capacity verified; no implementation certification implied. |
| R4: MHC dimensions/precision | **Partly resolved.** Matrix dimensions and FP32 parameter bytes fixed; operand/compute precision still unresolved (F3). Explicit vector exclusions are an improvement. |
| R5: Measurements and join | **Not resolved.** Annotations do not restrict rendering, and join operands remain inconsistent (F1/F4). |
| R6: Hardware provenance | **Substantially resolved for this nominal analysis.** Node/clock/reference distinctions repaired. Inherited RAM/topology are still not independently certified, but no full runtime-fit claim is made. |
| R7: Gate and workflow | **Partly resolved.** Reference-valued checks and emitted exclusions added; fail-open tracker and unchecked contracts remain (F5). |

The author explicitly defers the reusable skill's occupancy-formula alignment to the playbook-encoding phase. This re-review does not use that deferred documentation edit as a roofline blocker; it remains a tracked follow-up before that skill is relied upon for another analysis.

## Validation Performed

1. Ran the 13 current reference/conformance self-tests: all pass.
2. Re-fetched the pinned config and reference model; confirmed main-layer counts, FP32 compressor state, per-request indexer head weights, and MHC FP32 computation.
3. Regenerated the main report and the new [EMR join report](dsv4_roofline_vs_measured_emr.txt) in memory: exact matches. The older [join report](dsv4_roofline_vs_measured.txt) is not the current generator output; designate a canonical artifact or mark the older one historical.
4. Executed counterexamples for measurement invariance, missing rendered provenance, pool traffic, precision-independent compute selection, and missing-tracker behavior. Calculated join byte/FLOP discrepancies from the call contracts.
5. Inspected only benchmark argument/output contracts needed for byte accounting. No optimized kernel implementation was reviewed, no kernel or parity test was run, and no performance result was remeasured.

Production code, the response, and earlier reports are unchanged by this review.

## Next Acceptance Gate

Resolve F1-F5 with executable contract tests, regenerate the canonical artifacts, and provide a response distinguishing verified fixes from explicit exclusions. Keep the accepted R1/R3 corrections and the useful partial R2/R4/R6/R7 work. Once the revised roofline passes, implementation correctness and measurement validity can be reviewed as the next phase.

**This revision does not yet justify expert sign-off on the roofline, nor approval of implementation correctness or reported speedups.**