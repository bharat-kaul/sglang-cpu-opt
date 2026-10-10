# Assessment of the R7 Roofline Response

Reviewed response `6ce8394` and implementation `7cc4353`, on the clean response revision. Scope: aggregation validation and canonical-root launcher resolution from assessment `4ef6fbb`. This review used existing scratch inputs, in-memory CLI probes, and launcher-resolution blocks only. No native edits, performance runs, cluster submissions, or saved-input modifications.

## Remaining Finding

### R8-F1 (Medium): absent or incompatible provenance can still be written as VALIDATED

The [aggregator](../aggregate_roofline.py#L1) checks that each provenance object is truthy, then compares five fields using `.get()`: `slurm_job_id`, `head`, `threads`, `omp`, and `bind`. It does not require those fields to exist or contain valid values. Missing fields in every input therefore compare equal as `None`. Moreover, the producer's `torch`, `node`, and `peak_nominal` metadata are not included in compatibility checks.

Four independent probes through actual CLI `main()` with `--out` reproduced normal completion and an output artifact marked **VALIDATED**:

1. Three otherwise valid process inputs whose provenance contained **only** distinct `process_index` strings. No job, source, thread, or binding identity remained.
2. Three valid inputs with a different `torch` runtime string in the second record.
3. Three valid inputs with a different node in the second record.
4. Three valid inputs with different nominal BW/AMX/FP32 peak values in the second record. Aggregation still used its own hardcoded constants without rejecting the incompatible reference metadata.

The first probe is a direct bypass of the new-format identity requirement, not merely an optional metadata omission: unidentifiable measurements receive the same status as qualified producer records. The other probes show that listing provenance is still stronger than the compatibility enforcement. These are consumer fault injections; the historical job is not shown to have mixed or invalid measurements.

**Required correction:** validate each new-format provenance object against an explicit required schema before cross-record comparisons. Require meaningful, typed job/source/runtime/thread/binding/process identities; compare the complete declared compatibility contract, including reference-peak values, and bind those values to the constants actually used for aggregation. For this single-node replicated sweep, require the same node, or explicitly represent a separately qualified cross-node experiment instead of silently pooling it. Keep unstamped historical imports confined to the labelled legacy path. Add intended-reason, no-output-on-rejection CLI controls for missing/null/empty required identities and differing runtime/reference metadata, alongside a complete producer-shaped positive control.

This is the remaining portion of R7-F1, not a reopening of the count, duplicate-index, latency, or legacy-mode fixes below.

## Verified Closures

**R7-F2 CLOSED.** All four patched launcher resolution blocks now normalize both repository-root and repository-subdirectory submissions to the canonical root. A non-repository fallback FATAL-exits with status 1. The reviewer ran 12 combinations: four launchers x root/subdirectory/outside-repository. All behaved as intended, with the repository-relative benchmark path present after each valid resolution. This checkout uses a **`.git` file**, so the valid cases also exercise the worktree-style layout. Only the unchanged resolution blocks ran under `set -euo pipefail`, using `/dev/stdin` as a non-checkout script location to exercise the Slurm-style fallback; activation and benchmark bodies did not run.

**R7-F1 original counterexamples CLOSED; full identity contract PARTIAL.** Nine committed `--selftest` controls pass. Independent through-`main()` checks additionally confirmed **11 negative controls**, each rejecting for the intended reason with **zero output-file opens**: wrong count, unstamped inputs outside legacy mode, stamped inputs in legacy mode, conflicting job/source/thread metadata, duplicate process index, missing M coordinate, and NaN/+inf/zero/negative/boolean latency. Valid complete new-format and historical legacy inputs both produce output with their respective statuses. The positive and negative controls use intercepted file I/O; no evidence artifact is overwritten.

**Historical arithmetic and explicit legacy status verified.** The committed `aggregate(..., legacy=True)` reproduces the entire published `kernels` mapping at all **40 coordinates**, including samples, medians, ranges, floors, percentages, and regime tags. The published artifact is now explicitly `LEGACY_RECONSTRUCTED`, with historical run identity under `reconstructed_from_log`, not claimed as producer-stamped. Top-k M1 remains **0.67%**. Reconstruction of historical log metadata remains separate from generation of numerical rows.

## Scope and Status

R6-F1's nominal-reference wording closure and the R5-F1/R5-F2 correctness closures remain unchanged. This patch does not alter native kernels, replay validation, F4, or promotion policy, so those tests were not rerun. The response correctly separates the 19 persistent replay controls from the earlier reviewer's main-entry fault probes.

**Disposition:** close the launcher issue and retain all demonstrated aggregation repairs. Complete provenance-schema and compatibility enforcement before claiming prospective aggregation fully validated. No new performance run is needed. F4 remains **PARTIAL**, promotion **BLOCKED**, and thresholds **UNRATIFIED**; this assessment authorizes no budget, experiment, dispatch/E2E work, or phase promotion.