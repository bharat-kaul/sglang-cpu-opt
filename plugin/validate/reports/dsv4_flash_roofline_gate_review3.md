# DSv4 Flash Roofline Gate: Third Review

Date: 2026-10-08.

Reviewed revision: `b37420a8e495688c15f1fffd2db5d9cf60f0c9b3`.

Reviews the [second author response](dsv4_flash_roofline_gate_response2.md) against the [previous re-review](dsv4_flash_roofline_gate_rereview.md).

## Verdict

**FAIL: the earlier corrections are largely implemented, but three remaining issues prevent roofline sign-off. Implementation review remains deferred.**

All **22** current self-tests pass. Both canonical generated reports reproduce exactly. Measured coordinates are now restricted correctly; unsupported latency placeholders are removed; FP32 MHC uses its own nominal compute ceiling; and the four previously specified join byte formulas are corrected. These are substantive fixes, not merely response-document changes.

The remaining blockers are a wrong compute resource in the sparse-attention join, a mismatch between the indexer observation and its claimed benchmark contract, and incomplete tracker validation. No hardware calibration or kernel execution is needed to resolve them. Keep the agreed nominal-peak methodology and the explicitly partial scope of the analytical model.

## Verified Fixes

| Check | Result |
|---|---|
| Earlier R1/R3 and corrected matrix inventories | Preserved: 2/21/20 main-layer counts, 16,896 attention positions, FP8 shared experts, corrected compressor capacities and MHC K=16384 |
| Measured coordinates | All three measured rows return a number only at M=32; M=1/8/16/64 return N/A |
| Unsupported small-op observations | All five unmodeled rows return no numeric latency at any displayed M |
| FP32 pooling call contract, R=128/D=512/M=32 | 17,104,896 bytes, including KV, scores, APE, and output |
| Compute resources | BF16 AMX: 124.5184 TFLOP/s; FP32 AVX-512: 7.7824 TFLOP/s; MHC hc_fn now has FP32 operands and compute |
| Previously specified join contracts | Indexer BF16 hypothetical boundary: 9,052,160 bytes; pool: 17,104,896; Sinkhorn: 6,252; combine: 2,621,952 bytes and 917,504 FLOPs |
| Tracker absent, malformed JSON, or exact disposition BOGUS | Self-test fails, preventing normal main-report emission |
| Reference identifier | Correct centralized revision retained: `60d8d70770c6776ff598c94bb586a859a38244f1` |
| Publication cleanup | Old join marked historical; new EMR join is canonical; unsupported causal conclusions removed |

The pooling figure above verifies the declared standalone `kv, score, ape` boundary. It is not certification that this boundary includes the entire stateful compressor. Per-token state handling and postprocessing remain explicitly excluded. Similarly, corrected ideal targets do not certify the attached measured values.

## Remaining Findings

### G1. High: Sparse-Attention Join Still Applies the BF16 Ceiling to FP32 Work

The [sparse-attention join row](../dsv4_roofline_vs_measured.py#L57) describes a scalar observation and counts FP32 KV/query/output bytes, but supplies `PEAK`, the BF16 AMX ceiling, instead of `FP32_PEAK`.

The [cited benchmark's call contract](../bench_sparse_attend.py#L39) creates FP32 query/KV/sink tensors and distinguishes scalar and AMX calls and timings. The row specifically identifies the scalar observation. This review inspected that boundary and labeling, not the optimized kernel implementation.

For its declared M=32, H=64, K=512, D=512 QK/AV work:

| Quantity | Current join | Matching nominal FP32 QK/AV target |
|---|---:|---:|
| FLOPs | 2,147,483,648 | Same |
| Current counted bytes | 41,943,040 | Same for this controlled comparison |
| Compute reference | 124.5184 TFLOP/s | 7.7824 TFLOP/s |
| max(bytes/BW, FLOPs/peak) | 117.028571 us | 275.941053 us |
| Dominating branch | Bandwidth | Compute |
| 1,781 us observation / target | 15.2185x | 6.4543x |

This changes both the reported branch and roofline distance. The FP32 nominal target remains an optimistic arithmetic reference, not a prediction that a scalar implementation achieves vector peak. Softmax and other omitted work are not newly modeled by this comparison.

**Close when:** associate every joined row with an explicit compute dtype/resource and assert the association. For this scalar FP32 row, select the FP32 nominal ceiling. If the intended comparator is instead a prospective BF16 AMX algorithm, label it as a different target with explicit conversion/numerical assumptions and attach the appropriate observation; do not describe it as the same benchmark contract.

### G2. High: The Indexer Source Does Not Produce the Claimed Observation Contract

The [indexer join row](../dsv4_roofline_vs_measured.py#L35) cites [bench_indexer_logits.py](../bench_indexer_logits.py#L51) for a 0.536 ms observation at M=32, context=1024, with per-request BF16 keys/query. That benchmark actually lists only these cases:

- M=1, context=512;
- M=1, context=4096;
- M=8, context=2048.

It uses a shared key tensor `[S,D]`, not independent `[M,S,D]` keys. A filename alone therefore does not support the joined coordinate, ownership, or timing provenance.

The nearby [bench_idx_logits.py](../bench_idx_logits.py#L34) does exercise M=32/context=1024 with per-request keys. However, its public call inputs are FP32 `q[M,H,D]`, `kv[M,S,D]`, and weights. Any BF16 conversion/preparation inside that timed boundary cannot be silently replaced by a BF16 input contract. This inspection does not establish which path produced 0.536 ms; no raw run record is linked by the join.

At that alternative FP32 public boundary, counting each input once and FP32 output gives **17,965,056 bytes**, not **9,052,160 bytes** for the currently declared BF16 boundary. The main model's BF16 projected-query target can remain a valid declared target. The problem is asserting that the observed benchmark implements exactly that input boundary without evidence.

The response also says the main report carries benchmark provenance. [phaseA](../dsv4_roofline_p2.py#L393) stores neither source nor provenance in the emitted rows, and the footer refers to op notes that are not printed. Executing the generator confirms `bench_topk.py` is absent from its output. Sources remain available in code, but the report is not self-contained on this point.

**Close when:** link the actual result record and benchmark/kernel revision for each observation, not only a benchmark filename or upstream model SHA. For indexer, reconcile the observed shape, key ownership, public input dtype, preprocessing/timing boundary, and selected backend. Either count that boundary or label the measurement as a different-boundary observation compared with a prospective target. If the result cannot be sourced now, mark the observation unverified/unavailable rather than certified. Print provenance or an auditable result link in the canonical report. No remeasurement is required merely to make the report honest.

### G3. Medium: Tracker Validation Still Accepts Invalid and Incomplete Content

The new [load_tracker](../dsv4_roofline_p2.py#L284) correctly rejects missing files, malformed JSON, and a disposition exactly equal to BOGUS. However, disposition validation uses `startswith`, and required item fields and coverage are not enforced.

The following injected tracker variants all passed `load_tracker()` and the full self-test, then permitted `p0()` to emit the report:

| Injected content | Result |
|---|---|
| One item with disposition `MODELED_BOGUS` | Accepted |
| One item containing only `{"disposition": "MODELED"}` | Accepted despite no item identity or source |
| Current tracker with all EXPLICITLY-UNMODELED items removed | Accepted; tracker-only exclusions silently disappear |

These were in-memory test inputs; no tracker file was modified. The first two cases are invalid schema, not a question of whether every scientific omission can be detected automatically. The third shows that required, already-known exclusions can disappear without disposition or review. Locally annotated exclusions survive, but do not cover all the tracked gaps.

The current negative self-test reproduces part of the validator logic on a BOGUS string rather than exercising the loader with structurally invalid or incomplete input. It therefore misses these failure modes.

**Close when:** validate a bounded disposition enum, with any qualifier in a separate field or explicitly supported syntax; require identity and disposition-specific evidence/reason/plan fields; reject duplicate or missing required tracking identities. Maintain coverage for known modeled/measured/excluded categories, with deliberate dispositions when a gap closes. Exercise the actual loader and report-emission path with the invalid variants above. This does not require network access during self-test or an attempt to infer every possible missing operation.

## Disposition Against F1-F5

| Previous finding | Current disposition |
|---|---|
| F1: Single-point measurements | **Rendering behavior resolved. Provenance incomplete:** source selection and emitted evidence remain open under G2. |
| F2: Pool score stream and state precision | **Specified contract corrections verified.** FP32 streams/output and explicit remaining compressor exclusions are present; full compressor execution is not certified. |
| F3: Precision-to-compute selection | **MHC/helper corrections verified, join incomplete:** sparse-attention row still uses the wrong resource (G1). |
| F4: Join operand contracts | **Four requested formulas corrected, observation matching incomplete:** the cited indexer case/ownership/dtype does not match (G2). Causal conclusions were removed. |
| F5: Publication gate | **Partly resolved:** missing/malformed/basic-invalid files now fail; schema and coverage bypasses remain (G3). Reference typo and historical-artifact handling are fixed. |

This review does not reopen accepted R1/R3 corrections, demand measured ceilings instead of nominal ones, or make the deferred reusable-skill occupancy edit a blocker. Previously declared partial-model exclusions remain limitations, not proof that the whole model has a complete cost inventory.

## Validation Performed

1. Ran all 22 current conformance/contract self-tests successfully.
2. Independently checked all measured and unmodeled rows over M=1/8/16/32/64, pooling bytes, and precision-specific ceilings.
3. Regenerated [the main report](dsv4_roofline_emr.txt) and [canonical EMR join](dsv4_roofline_vs_measured_emr.txt) in memory; both match their checked-in artifacts exactly.
4. Tested missing/malformed/invalid and structurally incomplete trackers via in-memory injection; checked self-test results and report emission.
5. Recomputed the sparse-attention precision counterexample and checked the cited indexer/sparse benchmark call contracts. No kernel implementation audit, parity execution, performance remeasurement, or cluster job was performed.

For reproducing the remaining tracker issue with the actual loader, from the repository root:

```python
import contextlib
import importlib.util
import io
import json
from unittest.mock import mock_open, patch

spec = importlib.util.spec_from_file_location(
    "roofline", "plugin/validate/dsv4_roofline_p2.py"
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
probe = {
    "reference_revision": model.REF_REVISION,
    "items": [{"item": "probe", "disposition": "MODELED_BOGUS"}],
}
with patch("builtins.open", mock_open(read_data=json.dumps(probe))):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        passed = model.selftest()
        model.p0(model.load_tracker())
print("Invalid tracker passed self-test:", passed)
print("Report emitted:", "P0 CAPACITY" in output.getvalue())
```

At the reviewed revision both printed results are `True`.

## Next Gate

Resolve G1-G3, add the corresponding contract tests, and regenerate the canonical reports. If an observation cannot yet be traced, explicitly withhold it; do not invent provenance or rerun expensive jobs just to populate the table. Re-review the roofline before proceeding to implementation correctness or validating speedups.

Only this review document is changed by this review. Production code, author responses, and earlier reports are preserved.