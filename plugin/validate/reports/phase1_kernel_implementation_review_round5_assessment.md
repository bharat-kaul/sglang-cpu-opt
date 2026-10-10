# Assessment of the Round-5 Correctness Response

Reviewer: GitHub Copilot. Reviewed response: `020f429d83d3a029729f5f0a86666544c076ae4a`, implementation `0376948`, against [the round-4 assessment](phase1_kernel_implementation_review_round4_assessment.md). Runtime checks ran at `b51d1eb`; the intervening change only adds documentation to the standalone optimization report. No native kernel source changed in this response.

**Verdict: accept the demonstrated fixes to the original R4-F1/F2/F3 counterexamples. Two evidence-validation gaps remain; overall correctness qualification is still PARTIAL, and promotion is BLOCKED.** The failures below are controlled validator/builder fault injections, not observations that normal native outputs are wrong. The stock F4 run completes without hard failures.

## Findings

### R5-F1 - High: replay accepts nonfinite GPU reference output

[Archive validation](../test_sparse_cpu_vs_gpu.py#L35) now checks identity, tensor ranks, coordinate relationships, and positive finite scale. It does not reject nonfinite tensor contents, and [the replay comparison](../test_sparse_cpu_vs_gpu.py#L104) does not turn nonfinite reference/output metrics into a failure.

Reproduction used the safely loaded real job-384532 archive. Replacing only the N1 record's `gpu_out` with a same-shape NaN tensor was accepted by `validate_archive(..., expected_job="384532")`. The actual `main()` entry point then ran all four compiled paths, printed `nan` for both cosine and maximum error in every matched-output row, printed `>>> done`, and returned normally.

This is invalid oracle evidence, not an approximation exceeding an unratified tolerance. Reject invalid reference tensors before comparison and invalid candidate outputs before computing metrics. Validate the documented tensor/dtype/finite domain before conversions; any allowed mask values need an explicit field-specific policy. A sparse attention output containing NaN is not a valid mask case. Add persistent, intended-reason controls for nonfinite reference and candidate output alongside a finite positive control.

The archive substitutions were in memory only. The saved archive was not changed, and its genuine records were not found to contain this fault.

### R5-F2 - Medium: joint-coordinate checks validate a different build from the evaluated input

The [new preflight loop](../f4_acceptance.py#L534) calls each required case's builder and validates that result. Later, [evaluation](../f4_acceptance.py#L550) calls `run_case()`, which invokes the builder again. The required-coordinate predicate is not applied to the actual evaluation tuple. Checking one build does not establish the shape of later builds, especially for dynamic or stateful builders.

A full `run()` probe kept all 48 case declarations and used seeds 0/100. Only normal compressor `r8d128` received a wrapper: its preflight call returned the expected R8 tensors; its two evaluated calls returned consistently sliced R4 KV/score/APE tensors. Output shape remained `[8,128]`, and both native and local reference accepted that valid R4 operation. Observed window sequence was **[8,4,4]**; the full gate returned **0/PARTIAL, no hard failure**, under the R8 case label.

Bind required coordinates to the exact input tuple passed to candidate and reference on every evaluated seed. Build once for each evaluation, validate that tuple, then use it for both sides. Keep declaration-presence checks, but do not treat a separate preliminary sample as execution evidence. Add a builder-drift negative control that fails for the coordinate mismatch itself.

This does not undo the corrected removal checks: missing heavy indexer M64 and missing compressor R8/D128 are now rejected. The remaining issue is the stronger claim that the required shapes were actually evaluated. Current fixed-shape stock builders were not shown to drift.

## Scoped Closures

| Previous finding | Verified disposition |
|---|---|
| R4-F1 canonical job identity | **Closed for the reported cases.** The real archive accepts requested `slurm_job_id=384532`; a wrong requested identity and conflicting legacy alias reject for the expected reasons. |
| R4-F2 mislabeled N / shape relationships | **Closed for the reported coordinate failures.** Real N1 tensors labeled N64 are rejected; persistent tests also reject inconsistent output H/D and nonpositive scale. Complete evidence validity remains open under R5-F1. |
| R4-F3 removal of required joint coordinates | **Closed for the original removals.** Full-run controls reject omission of heavy-tailed indexer M64 and compressor R8/D128. Actual evaluated-coordinate binding remains incomplete under R5-F2. |
| Persistent replay selftest | **Added and working:** 12 controls pass without a kernel build. Its positive fixture is synthetic and producer-shaped; it is not a GPU capture. This review separately checked the genuine saved archive. |
| Manifest claim drift | The targeted manifest comment now says screening against a proposed, unratified threshold rather than GPU-ratified acceptance or an intrinsic noise floor. |

Prior scoped A1-A3 closures remain in force. No requirement for exact sparse GPU equality, new error tolerance, dispatch change, or performance optimization is introduced by these findings.

## Budget and Remaining Scope

The [expanded methodology](../results/kernel_opt_queue.json#L7) adopts the requested separation of workload/boundaries, structured residual perturbations, independently chosen downstream criteria, frozen validation, and a bounded pre-submit plan. It correctly retains option (a), distinguishes evidence collection from threshold approval, and records that 0.00195 is not a universal sparse error bound.

This is a requirements checklist for an experiment, not yet an executable, authorized experiment plan: it says to name the workload, metrics, correlations, resource bounds, and dispositions rather than supplying all of their concrete values. Accept the methodological correction, but do not ratify a numerical budget or authorize a launch from it. Other continuous-family budgets remain separate obligations.

Expanded saved GPU cases, broader non-sparse all-M and sparse K/head/layout/state coverage, and existing timing-identity/comparator caveats remain explicitly pending as the response states. They are not new defects in this delta. Standalone optimization opportunities and deferred dispatch/end-to-end work remain governed by the separate [standalone review](standalone_kernel_optimization_review.md).

## Verification

- Replay selftest: **12 checks pass**.
- F4 selftest: **26 checks pass**, including the two new joint-coordinate removal controls.
- Full stock F4 run: **48 cases x 5 seeds = 240 evaluations**, return 0/PARTIAL, no hard failures.
- Genuine archive: expected source SHA-256 and requested canonical job validated; real records are N1/8/64. Wrong requested job, conflicting alias, and inconsistent N rejected for their intended reasons.
- Actual compiled replay with a NaN-reference substitution reproduced R5-F1; a full-run builder-drift probe with seeds 0/100 reproduced R5-F2.
- Promotion selftest: **5 checks pass**; live decision given the independently observed PARTIAL result is **BLOCKED**.

Runtime checks used `/scratch/bkaul/venvs/sglang-cpu/bin/python -B`, OMP_NUM_THREADS=1 and MAX_JOBS=2. No production code, policy, queue, saved archive, or calibration output was edited. No cluster/GPU jobs, model loads, performance runs, or dispatch changes were made. Only this assessment is authored by the review.