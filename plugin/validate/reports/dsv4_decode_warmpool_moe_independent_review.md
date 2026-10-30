# Independent review: warm-pool configuration and MoE optimization for B <= 64

Reviewed 2026-10-28 against repository revision
`a69fb9ac999600b6a166d9c9ccc90b281d40b91b`.

Subject: [warm-pool/MoE pivot assessment](dsv4_decode_warmpool_moe_pivot_assessment.md).

## Decision

**Support the pivot toward MoE and defer the monolithic layer rewrite.**
The warm-pool configuration has credible evidence of a substantial decode win.
However, correct the absolute rates, verify the actual native ISA dispatch,
and investigate the observed prefill warning before promoting it as a default.

The objective remains maximum correct decode performance and useful bandwidth
efficiency on one EMR socket for batches 1 through 64. Fusion is a means, not
the objective.

This review used the report, saved samples, launchers, timing code and installed
SGLang native sources. Arithmetic was independently reconstructed; no new
full-model performance run was launched.

## 1. Warm-pool gain: credible relative improvement, incorrect absolute rates

The settings are established before worker startup, unlike the previously
invalid client-environment A/B. A separate attribution run also observes a
large decode improvement.

But [validate_decode_threads_clean.py](../validate_decode_threads_clean.py)
subtracts a one-output-token request from a six-output-token request and divides
by six. Normally that difference contains five subsequent decode steps.

Recomputed from the saved samples, conditional on every request producing six
tokens:

| Batch | Cold | Warm | Relative improvement |
|---|---:|---:|---:|
| 1 | about 602 ms/step, 1.66 tok/s | about 242 ms/step, 4.12 tok/s | 2.48x |
| 32 | about 3228 ms/step, 9.91 tok/s | about 1476 ms/step, 21.68 tok/s | 2.19x |

Sources:
[cold samples](../results/dsv4_warmpool_cold_394560.json),
[warm samples](../results/dsv4_warmpool_warm_394561.json).

The [independent attribution log](../results/dsv4_attr_warmcold_394565.log)
reports approximately 1.469 s/step and 21.78 tok/s for warm B=32, closely
matching the corrected estimate.

Retain the relative win; correct the absolute rates. Prefer worker-step timings
and actual output counts over subtracting a separately measured prefill.
Six requests in one process are not six independent process replications.
Add an order-balanced process repeat before generalizing the result.

## 2. Prefill has an existing warning, not just an unmeasured risk

The raw files `/scratch/bkaul/dsv4_attr_cold_394565.json` and
`/scratch/bkaul/dsv4_attr_warm_394565.json` contain accumulated prefill layer
times of 14.60 s and 72.70 s respectively: warm is approximately 4.98x higher.

These totals combine warmup and benchmark passes. They do not independently
establish steady-state TTFT regression, but they contradict any unconditional
"strictly better on a dedicated socket" conclusion.

The [sticky-cap implementation](../../intel_cpu_models/_dsv4_cpu_infra.py)
leaves the thread count reduced after decode. Subsequent prefills inherit that
count unless another path changes it.

Measure first prefill, subsequent prefill, sustained decode and mixed
prefill/decode separately. Identify the loaded OpenMP runtime; compare immediate
parking, finite spin duration and infinite spin. Record effective worker thread
counts, affinity and memory placement, not the client's thread count.

For the scoped single-socket TP=1 objective, TP>1 testing is not a prerequisite.
Keep promotion explicitly scoped rather than expanding the campaign.

## 3. Native MXFP4 execution does not prove AMX execution

The [diagnostic log](../results/dsv4_moe_diag_394490.log) establishes packed
weights, shapes, occupancy and thread counts. It contains no proof that AMX
instructions executed.

The installed native source:

```text
/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang/kernels/aot/csrc/cpu/moe_fp8.cpp
/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang/kernels/aot/csrc/cpu/gemm.h
```

selects BRGEMM using:

```text
avg_M = max(1, batch * topk / total_experts)
use_brgemm = avg_M > 4
```

With top-6 and 256 experts, every batch from 1 through 64 yields `avg_M=1`.
That source selects the non-BRGEMM path throughout the target range.

**Verify the loaded binary and instrument the actual branch.** Installed source
alone is not binary provenance, but it directly challenges the explanation of
"underfilled AMX tiles." Do not infer the ISA from an operator/backend name.

The B=32 diagnostic records rows/expert min/mean/max of **1/2.1/10**, not
uniformly 1-6. Capture distributions across layers and steps; the first observed
sample is not a whole-model routing census.

## 4. MoE is the right priority; the floor comparison needs repair

The measured expert-apply share is supported:

```text
22.9314 s / 31.9681 s = 71.7%
```

The claimed 44 ms per apply appears misnormalized. The attribution launcher
runs warmup and benchmark, each with 11 decode steps. Assuming one expert apply
per layer per step:

```text
22.9314 s / (2 * 11 * 43) = 24.24 ms/apply
```

Confirm using recorded call counts instead of reconstruction.

Rebuild the claimed 6-10x gap using per-layer active expert counts, packed
weights plus scales, repeated reads/conversion work, actual memory placement,
configuration-matched bandwidth and the compute ceiling of the executed path.

A gap to ideal weight-streaming bandwidth does not uniquely identify dispatch.
Conversion throughput, vector computation, insufficient memory concurrency and
repeated reads can all contribute.

## 5. Defer the monolith without asserting a 12% hard ceiling

The attribution sums `mhc.sinkhorn` and `mhc.combine` alongside their enclosing
`mhc.hc_pre`. Removing just this nested double count raises the warm residual
from about 12.28% to 12.92%.

More importantly, fusion can reduce costs inside timed operators: scratch
allocation, conversion, intermediate writes and internal parallel regions.
"MoE is one kernel" does not make these costs irreducible.

The justified decision is simpler: **MoE dominates current measured layer time,
so targeted MoE work has higher initial priority than a broad layer rewrite.**

## 6. Recommended path for maximum decode performance

### Priority 1: establish the warm operating envelope

Carry B=1, 2, 4, 8, 16, 32 and 64 with representative context lengths.
Screen thread counts under warm settings, including counts above eight;
eight is not an established warm optimum.

Use short screening runs and replicate the promising configurations. Preserve
decode-step latency, throughput, TTFT and mixed-workload metrics separately.
Retain raw samples, worker settings, source/build identity and actual batch/
token counts. Confirm numerical parity on matched inputs and state.

### Priority 2: dispatch by actual expert occupancy

| Actual expert group | Candidate strategy to measure |
|---|---|
| One/few rows | Packed-native vector microkernel with low setup and rereads |
| Larger groups | BRGEMM/AMX where conversion and tile setup amortize |
| Mixed groups | Per-group/block choice instead of a diluted global average |

Do not force AMX everywhere. Do not let an average over all 256 experts force
hot groups onto tiny-vector kernels either.

Grouping experts for scheduling is useful. Combining unrelated expert rows
into one ordinary GEMM is not equivalent: their weights differ.

### Priority 3: reduce native MoE setup and data movement

The installed `moe.cpp` allocates routing/scratch buffers per call and has
multiple execution stages. Profile and test:

- reusable, concurrency-safe scratch;
- compact active-expert routing metadata;
- expert/output-block scheduling that reuses packed panels;
- one-time packing of weights and scales;
- in-kernel cache-local dequantization, not whole-expert widening;
- compatible activation/weighting/store fusion;
- fewer parallel-region transitions within MoE.

Separate routing, allocation, conversion and arithmetic costs. Optimize the
costs demonstrated to matter, using captured real occupancy and operand shapes.

### Priority 4: reattribute after each major improvement

If the present 72% share remains representative:

| Hypothetical MoE speedup | Conditional overall speedup |
|---|---:|
| 2x | 1.56x |
| 3x | 1.92x |
| 4x | 2.17x |

These are Amdahl projections, not measured promises. Attention, dense
projections and remaining glue rise in priority as MoE improves or context grows.

## Next decisive experiment

Combine:

1. Worker-step timing with correct output accounting.
2. Warm thread-count screening over B <= 64.
3. Actual vector/BRGEMM branch counters and per-expert occupancy.
4. Matched-input tests of existing vector and BRGEMM paths on observed groups.

Keep the warm-pool result as a strong scoped finding, not a universal default.
The best current route toward useful socket bandwidth efficiency is correct
dispatch and efficient packed-weight execution across real expert occupancies,
supported by a stable runtime configuration.
