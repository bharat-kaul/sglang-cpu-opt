# Assessment of the Round-4 Kernel Response

Reviewer: GitHub Copilot. Reviewed [the response](phase1_kernel_implementation_review_round4_response.md) at `38b9484acd4260b8f83f9c752593ef07349e3157`, including implementation commit `29c6a79`, against [the preceding assessment](phase1_kernel_implementation_review_round3_response_assessment.md) (`90aa669`). Worktree clean before review/report creation; no native kernel source changed.

**Verdict: A1-A3 are closed for the reported counterexamples. A4/A5 are partially closed, not completely fixed. Phase 1 remains PARTIAL; promotion remains BLOCKED.** The response makes substantial, verified progress. The remaining findings concern actual evidence identity and coverage, not a demand for exact sparse GPU equality or a new numerical tolerance.

## Findings

### R4-F1 - Medium: optional replay job binding reads the wrong field

The [new job check](../test_sparse_cpu_vs_gpu.py#L51) reads `prov.get("job")`, but the [oracle producer](../test_sparse_gpu_oracle.py#L81) saves `slurm_job_id`.

Using the safely loaded real job-384532 archive, replay without a requested job succeeds. Requesting its correct identity, `ORACLE_JOB=384532`, instead fails with `saved io job None != expected run identity '384532'`. Conversely, adding `job="other-run"` while retaining the actual `slurm_job_id="384532"` makes a request for `other-run` succeed. These checks exercised `main()` with the module's expected-job value patched in memory, equivalent to its import-time environment setting; no archive was modified.

Use the producer's canonical `slurm_job_id` field. Test both a correct requested identity and an incorrect one against the producer-shaped archive. If legacy aliases are supported, conflicting fields must be rejected, not allowed to override the canonical identity.

### R4-F2 - High: replay coordinates are checked for presence, not consistency

The [record validation](../test_sparse_cpu_vs_gpu.py#L56) rejects empty records and missing keys, which closes the previous empty/missing-coordinate counterexamples. It does not validate coordinate values against the tensors being compared.

Changing only the saved N1 record's `N` field to 64 is accepted. The script prints `64 scalar bf16-matched ...` and all four matched-path rows, although Q/KV/output still have batch size 1. Native dispatch follows the tensor shape: this is M1 execution reported as M64, not merely a cosmetic label. Such records cannot substantiate the requested M sweep.

Validate at least `N == q.shape[0] == kv.shape[0] == gpu_out.shape[0]`, the complete Q/KV/output/sink shape relationships, and the documented tensor/dtype/scale domain before comparison. Keep the requested case inventory separate from individual-record validation; a nonempty subset must not silently become a complete qualification claim.

### R4-F3 - High: required distribution and compressor-shape coverage remains removable

The [new inventory checks](../f4_acceptance.py#L497) correctly enforce BF16-KV path presence, the masked-compressor kind, and every shipped sparse path's five M labels. These are useful scoped closures.

They do not yet enforce the independently required joint coordinates. Full `run()` probes with the normal native candidates and calibration/held-out seeds 0/100 still return 0/PARTIAL after either independent removal:

| Removed required coverage | Remaining cases | Result |
|---|---:|---|
| Heavy-tailed indexer M64 | 47 | No hard failure |
| Both normal/heavy compressor R8/D128 cases | 46 | No hard failure |

The first leaves normal indexer M64, so separate path/M presence checks remain satisfied. The second leaves other continuous compressor shapes and the masked case. This is the same previously required distribution/three-shape scope, not a newly added qualification requirement.

Declare and enforce required tuples of op/path/M/distribution/shape, with actual built tensors checked against their declarations. The broader all-M coverage gaps for non-sparse families and sparse K/head/layout/state variants documented in round 3 are not repaired by this sparse-M extension. Close the specific removal controls, not the entire required-inventory finding.

## Verified Closures

| Earlier finding | Current disposition and evidence |
|---|---|
| A1 valid `-inf` masks | **Closed, scoped.** Valid masked top-k positive control passes; NaN is still rejected. The helper rejects NaN/+inf while permitting the declared negative-infinity domain. This does not certify the full all-masked/sentinel caller workflow. |
| A2 composed operand/order mismatch | **Closed, scoped.** Runtime taps confirm the two primary candidate calls receive BF16-rounded Q/KV, and the reference gather exactly matches published `sorted=True` top-k ordering. Candidate order is preserved. A third original-FP32 call is separately recorded as `metric_fp32in_diag`. Seed-0 composed screening passes with max absolute error 0.001953125; that is an observation, not a ratified tolerance. |
| A3 composed repeat/reference dtype | **Closed for the reported failures.** Equal-valued FP64 references and composed repeat dtype flips now fail for the expected contract reasons. `_same_contract` includes dtype, shape, and device. Layout and complete caller contracts remain outside this scoped closure. |
| A4 replay identity/inventory | **Partial.** Restricted loading, expected source hash, empty-record rejection, and required-key checks work. Optional job identity and record coordinate consistency remain defective (R4-F1/F2). |
| A5 coordinate removal | **Partial.** Removing BF16-KV, masked pooling, or required sparse M16 now fails; scalar/FP32-BMM/best-of cover M1/8/16/32/64. Required joint coordinates remain incomplete (R4-F3). |
| Targeted claim-drift corrections | The cited screen comment, sparse conformance metadata, and budget note no longer describe observed discrepancies as a proven op-order minimum or the active reference as globally normalized. The separate manifest comment still uses legacy "GPU-ratified"/"own noise floor" language; it is not acceptance authority. |

Source-hash replay controls now reject both a wrong hash and a missing hash even with `validated=True`. The current archive succeeds with no requested job, empty records fail, and a record missing KV fails before comparisons. Therefore the previous wrong-hash/empty-archive findings should not be carried forward unchanged.

The response's opening claim of a dedicated typed control for every finding is broader than the implementation: the F4 suite contains the new A1/A3 and removal controls, but no persistent replay self-test for A4 is supplied. Reviewer-run positive and negative replay controls exposed R4-F1/F2. Preserve those controls next to the replay validator.

## Budget Proposal

**Recommendation: choose option (a), a bounded pre-sign-off qualification experiment, after explicit human approval of its plan. Do not choose option (b) merely to unblock promotion.** Closing Phase 1 on screening would change the agreed acceptance dependency; it would not establish numerical correctness. This review does not authorize an experiment, ratify a budget, or amend that dependency.

The calibration/untouched-validation separation and dummy-structural versus real-weight-sensitivity distinction are correct procedural improvements. However, "inject an epsilon-scaled perturbation" is not yet a complete experimental design or numerical budget:

1. Name the deployment workload, captured input/caller boundaries, model revision, dtypes, TP setup, and reference/candidate identities. Include the required M, K, masks, and layer/state regimes, with explicit proxy limitations.
2. Specify perturbation construction, norm/scaling, sign/direction, and correlations across channels, tokens, layers, and steps. Replay measured candidate residual tensors where possible and include justified stress directions; one generic perturbation magnitude cannot characterize systematic errors.
3. Define downstream/task metrics and acceptance criteria independently of the candidate's observed error. Use calibration to propose a candidate envelope; freeze it before untouched validation. Dummy propagation does not bound real-weight sensitivity. State how single-op results combine when multiple approximate kernels are active.
4. Provide a bounded compute/data plan, all open hypotheses, required captures, and a disposition matrix before any expensive launch. Persist expanded GPU-oracle I/O for candidate replay and retain run identity. Keep policy, coverage, performance-floor, and full-model gates separate.

The proposal's sparse residual of approximately 0.00195 is not a general upper bound. The current full F4 run's calibration summary already reports **0.00390625** for FP32-BMM and best-of at M16, against the blockwise reference with matched boundaries. No native source changed. This is evidence that more coverage can reveal a larger error, not evidence to ratify either 0.00195 or 0.004. Candidate-vs-replica, replica-vs-GPU, and composed-ordering discrepancies also must not be conflated into one universal epsilon.

Other continuous families' pending downstream budgets and the earlier performance/caller obligations remain in force. Approval to collect evidence is not approval of the resulting threshold or promotion.

## Verification

- F4 selftest: **24 checks pass**, including the new positive, contract-negative, and inventory-removal controls.
- Full unchanged F4 run: **48 cases x 5 seeds = 240 evaluations**, return 0/PARTIAL, no hard failures. The three shipped sparse paths now span the requested five M values; experimental AMX remains M1/8 diagnostic coverage.
- Runtime composed-path taps verify matched primary operands, published sorted reference gathering, and separately labelled original-FP32 diagnostics.
- Two full-run inventory-removal probes with seeds 0/100 reproduce R4-F3.
- Eight replay scenarios cover the genuine archive, correct/incorrect requested identities, wrong/missing source hash, empty inventory, missing KV, and inconsistent N metadata. They use restricted archive loading and in-memory substitutions only.
- Promotion selftest: **5 checks pass**. Calling the live decision with the independently observed PARTIAL result at clean `38b9484` returns **BLOCKED**: unratified policy, pending obligations, no human approval. This verifies the current blocked state, not hypothetical future promotion completeness.

All runtime work used `/scratch/bkaul/venvs/sglang-cpu/bin/python -B`, OMP_NUM_THREADS=1, and MAX_JOBS=2 where native extensions were loaded. No native source, author record, acceptance policy, queue, saved archive, or calibration output was modified. No cluster/GPU jobs, full-model loads, or new performance measurements were launched. Existing performance aggregation closures and native optimization opportunities remain as documented in round 3; they were not remeasured here.